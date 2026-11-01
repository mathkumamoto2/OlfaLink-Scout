"""Create Japanese PDF only from measurements actually produced on this computer.

Usage: python -m scout_genesis.report --account YOUR_OMNICAMPUS_ID
No synthetic values are ever written to a submission PDF.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


def measured_data(output: Path) -> tuple[dict, dict]:
    smoke = output / "smoke_result.json"
    evaluation = output / "eval_summary.json"
    for path in (smoke, evaluation):
        if not path.is_file():
            raise FileNotFoundError(f"実測ファイルがありません: {path}。先にsmoke/evaluateを実行してください。")
    data_smoke = json.loads(smoke.read_text(encoding="utf-8"))
    data_eval = json.loads(evaluation.read_text(encoding="utf-8"))
    for key in ("ppo", "random"):
        if key not in data_eval or data_eval[key].get("episodes", 0) <= 0:
            raise ValueError(f"評価データが不正です: {key}")
    if data_smoke.get("forward_progress_m", 0) <= 0.01 or abs(data_smoke.get("turn_yaw_change_rad", 0)) <= 0.03:
        raise ValueError("物理テスト不合格。レポートを生成しません。")
    return data_smoke, data_eval


def safe_id(account: str) -> str:
    """Only an account identifier, not a personal name or system path."""
    value = re.sub(r"[^A-Za-z0-9_-]", "_", account.strip())[:50]
    if not value or value in {".", ".."}:
        raise ValueError("--accountにはOmniCampusアカウントIDを指定してください")
    return value


def generate_report(output: Path, account: str) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    smoke, evaluation = measured_data(output)
    font = "HeiseiKakuGo-W5"
    pdfmetrics.registerFont(UnicodeCIDFont(font))
    normal = ParagraphStyle("BodyJP", fontName=font, fontSize=10, leading=17, spaceAfter=10)
    heading = ParagraphStyle("SectionJP", parent=normal, fontSize=13, leading=20, spaceBefore=10, spaceAfter=7)
    title = ParagraphStyle("TitleJP", parent=normal, fontSize=19, leading=27, alignment=TA_CENTER, spaceAfter=18)
    small = ParagraphStyle("NoteJP", parent=normal, fontSize=9, leading=15)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"説明資料_PAI最終課題_{safe_id(account)}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=43, rightMargin=43,
                            topMargin=44, bottomMargin=43, title="PAI最終課題 OlfaLink Scout", author="")
    story = [Paragraph("OlfaLink Scout", title),
             Paragraph("Genesisによる差動二輪ロボットの自律におい源探索とPPO強化学習", normal),
             Paragraph("1. 目的", heading),
             Paragraph("においセンサーの数値が得られても、発生源の位置は自明ではない。ロボットが局所観測を繰り返し、障害物を避けて発生源を探索する行動方策を学習することを目指した。", normal),
             Paragraph("2. 技術構成・工夫", heading),
             Paragraph("Genesis WorldでURDFの差動二輪ロボットと床・壁・障害物の剛体接触を表現する。Gymnasiumで環境を実装し、Stable-Baselines3のPPOで離散的な左右車輪の速度指令を学習する。発生源座標は方策の観測に含めず、観測15次元を使用した。", normal),
             Paragraph("仮想のにおい強度は気流方向に伸びた2Dガウス分布とし、ノイズと応答遅延を追加した。これは現実のガス拡散やppm測定を再現するものではない。正解発生源の位置は学習報酬と評価指標の算出にのみ使用した。", normal),
             Paragraph("3. Genesis物理テストの実測結果", heading),
             Paragraph(f"前進距離：{smoke['forward_progress_m']:.3f} m　／　旋回角度：{smoke['turn_yaw_change_rad']:.3f} rad", normal),
             Paragraph("4. 方策の比較実験（実測結果）", heading)]
    data = [["比較項目", "ランダム探索", "PPO"]]
    r = evaluation["random"]; p = evaluation["ppo"]
    data += [
        ["評価エピソード数", str(r["episodes"]), str(p["episodes"])],
        ["発見回数", str(r["successes"]), str(p["successes"])],
        ["発見率", f"{100*r['success_rate']:.1f}%", f"{100*p['success_rate']:.1f}%"],
        ["平均ステップ数", f"{r['mean_steps']:.1f}", f"{p['mean_steps']:.1f}"],
        ["終了時平均距離", f"{r['mean_final_distance_m']:.3f} m", f"{p['mean_final_distance_m']:.3f} m"],
    ]
    table = Table(data, colWidths=[185, 135, 135], hAlign="CENTER")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6ECF2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#C9D5E0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([table, Spacer(1, 12), Paragraph("5. 結果の解釈・限界", heading)])
    if p['success_rate'] > r['success_rate']:
        interpretation = "今回の試行では、PPOの発見率がランダム探索を上回った。ただし少数の評価エピソードでは偶然変動の影響があり、追加検証が必要である。"
    elif p['success_rate'] < r['success_rate']:
        interpretation = "今回の試行では、PPOの発見率はランダム探索を下回った。観測情報、報酬設計、学習時間を見直す必要がある。"
    else:
        interpretation = "今回の試行では、PPOとランダム探索の発見率が同じだった。この結果だけから性能改善を主張することはできない。"
    story.append(Paragraph(interpretation, normal))
    story.append(Paragraph("本実験は仮想濃度分布と簡易距離センサーを使用したシミュレーションである。気体の流体計算、実機センサー、未知の現実環境への転移は未検証であり、今後の課題である。", normal))
    story.append(Paragraph("※ 数値は同じPCで生成した outputs/smoke_result.json と outputs/eval_summary.json から自動取得。成果物の提出には実際の動画とGitHubコードURLを別途添付する。", small))
    doc.build(story)
    return path


def main():
    parser = argparse.ArgumentParser(description="実測結果からPAI最終課題PDFを生成")
    parser.add_argument("--account", required=True, help="OmniCampusアカウントID（氏名不可）")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    print(f"[OK] PDF: {generate_report(args.output, args.account)}")


if __name__ == "__main__":
    main()
