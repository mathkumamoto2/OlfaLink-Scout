# OlfaLink Scout — Windows Desktop / Genesis World / PPO

**目的：** 仮想のにおい源を探す差動二輪ロボットを、Genesisの剛体物理＋Gymnasium＋PPOで動かす。

**成果物の正確な位置づけ：** 本ZIPはWindows用に整備した**実行可能なソース一式**です。Pythonコードとモックテストは検証済みですが、制作環境に `genesis-world` を導入できなかったため、Windows上の実Genesis描画・車輪接触・PPOの収束は**未検証**です。起動後に `smoke` の成功を確認してください。学習や成功率を実測したという主張はしていません。

## 1. 使うPC

- **Windows 10/11 x64**。PowerShellでも、ZIPを展開してバッチをダブルクリックしてもよい。
- **Python 3.11 x64**（[python.org](https://www.python.org/downloads/)）。既に別バージョンが入っていても `py -3.11` を使用。
- 空きディスク容量：Python/PyTorch/Genesisのインストール用に**数GB以上**を確保。
- インターネット：**最初のインストール時だけ必須**。学習・推論にクラウドAPIは不要。
- まずは**CPU**で動かす。CPUの処理が遅い場合はNVIDIA GPU利用を検討するが、CUDA/PyTorchの適合確認が必要。
- Windowsに描画が対応したGenesis World v1.4 系の仕様を前提としている。GPUドライバとグラフィック環境はマシン依存。

## 2. 一番簡単な方法（番号順にダブルクリック）

0. このZIPを日本語・空白・OneDrive同期が含まれない短いパス（例 `C:\Projects\OlfaLink_Scout_Windows_Desktop`）へ展開する。
1. **`setup_windows.bat`**：仮想環境の作成、PyTorch CPU版、Genesis、ライブラリの導入、Python自動テスト。インストールにはネット接続が必要。
2. **`01_physics_test.bat`**：物理エンジンで車輪を回して、**前進と旋回**が起こることを検証。JSONを出力。失敗したら**次に進まない**。
3. **`06_preview_3d_scripted.bat`**（任意）：学習前の3D録画テスト。これは**固定の車輪指令**でありAIではない。
4. **`02_train_ppo.bat`**：PPOを20,000ステップ学習。`outputs/ppo_genesis.zip` ができる。20,000ステップだけで成功率が十分高くなる保証はない。
5. **`03_evaluate.bat`**：同じ開始シードでランダム方策とPPOをそれぞれ20試行し、成功率・距離・ステップ数を保存。
6. **`04_record_3d_ppo.bat`**：学習済みのPPOで約64秒のGenesis 3D動画を録画する。到達・時間切れで複数エピソードに分かれる可能性がある。**未学習時には生成できない**。
7. **`05_trajectory_only.bat`**（任意）：3D描画不要で2D軌跡のPNGを書き出す（**固定動作**）。

出力はすべて `outputs/` に生成される。モデル、動画、CSV、PNGは初めから同梱されていないため、他のPCの実測結果と混同しない。

## 3. PowerShellから個別実行したい場合

展開先でPowerShellを開いて以下を実行：

```powershell
.\setup_windows.bat
.\.venv\Scripts\python.exe -m scout_genesis smoke --backend cpu
.\.venv\Scripts\python.exe -m scout_genesis demo --backend cpu --policy scripted --viewer
.\.venv\Scripts\python.exe -m scout_genesis train --backend cpu --steps 20000
.\.venv\Scripts\python.exe -m scout_genesis evaluate --backend cpu --episodes 20
.\.venv\Scripts\python.exe -m scout_genesis demo --backend cpu --policy ppo --record
```

`--viewer` でPC上にGenesisのライブ3Dウィンドウを開く。Windows側のGPUドライバやOpenGLの状態によっては描画エラーが起こるため、まずは通常のヘッドレス物理テストを完了させる。

PPOの動作が悪い場合は、成功率を確認したうえで試行回数を増やす：

```powershell
.\.venv\Scripts\python.exe -m scout_genesis train --backend cpu --steps 100000
```

これは**ゼロからの再学習**であり、既存モデルへの継続学習ではない。CPUでの所要時間はPC性能に大きく依存する。

## 4. 研究的な設計

| モジュール | 内容 |
|---|---|
| Genesis World | URDF差動二輪ロボット、剛体・接触、床・壁・障害物、モーター速度制御 |
| 仮想におい | 風向で回転させた2Dガウス強度（単位なし、CFD/ppmではない）、加算雑音＋一次遅れ |
| 仮想距離センサー | 5方向の解析的レイキャスト（**Genesis LiDARではない**） |
| PPO | Gymnasium環境とSB3による学習。発生源座標は**観測に含めない** |
| 学習報酬 | 発生源までの真の距離を**学習時だけ**使う。実機で使用可能な報酬とは限らない |
| 動画 | Genesisカメラで直接MP4を録画。朱色の球は説明用の目標マーカーで方策には入力しない |

**状態**：位置・向き・におい強度・強度差・5方向の距離・前回行動・経過時間・速度など15次元。

**行動**：停止、前進、左旋回、右旋回、後退の5種類。元の2Dモデルの重みは**互換性がない**ので使用しない。

**実装上の注意**：ロボットの姿勢・車輪速度はGenesisによる。`reset()`時のみ配置と速度を初期化する（学習ステップ中に座標を直接書き換えて進ませることはない）。

## 5. 成果物と評価の読み方

- `outputs/smoke_result.json`：物理テストの前進距離（m）、旋回角度（rad）
- `outputs/ppo_genesis.zip`：学習済み方策
- `outputs/training_monitor.csv.monitor.csv`：学習中のエピソード記録（Monitorが拡張子を自動付加）
- `outputs/eval.csv`, `outputs/eval_summary.json`：PPOとランダム探索の**実測比較**
- `outputs/demo_trace.csv`, `outputs/demo_trajectory.png`：軌跡と仮想濃度マップ
- `outputs/genesis_3d_demo.mp4`：`--record`時のみ生成されるGenesisカメラ録画
- `outputs/demo_metadata.json`：動画の方策が`ppo`か`scripted`か、学習済みか、録画時間を保存

**合格や優秀賞はコードの提出だけでは判断できない。** Genesisの実際の動作動画と正しい結果、説明PDF、GitHub公開リンクが必要。今回の2D仮想におい場と実環境での嗅覚探索は異なる。発生源の推定成功率が高くても実ガスでの性能は保証しない。

## 6. 主なエラーと対処

- `'py' is not recognized`：Python 3.11 x64をインストールし、コマンドラインを再起動。
- `No suitable Python runtime found`：Python 3.11を選択してインストール（Windowsアプリストアのエイリアスには注意）。
- `ModuleNotFoundError: genesis`：`setup_windows.bat`をやり直す。`.venv`のPythonを使用しているか確認。
- `Microsoft Visual C++ ... DLL load failed`：最新の**Microsoft Visual C++ 再頒布可能パッケージ x64**、GPUドライバを確認。
- `CUDA`：最初は`--backend cpu`を使用。`--backend cuda`は対応するNVIDIA GPU・CUDA対応PyTorchがある場合に限る。
- `OpenGL / viewer / rasterizer`：グラフィックドライバ更新後に `smoke` を再試行。3D録画だけ失敗した場合は `05_trajectory_only.bat` で軌跡画像を取得。
- `Forward motion failed` / `Wheel turning failed`：ロボットのURDF・床接触・摩擦・車輪方向などの物理条件不整合。**この場合、学習しない**。
- PPOがランダムより悪い：不具合と断定しない。報酬設計・学習ステップ不足・局所観測の弱さなどを検討し、実測を記録する。

### 公式資料

- Genesis Installation: https://genesis-world.readthedocs.io/en/v1.4.0/user_guide/overview/installation.html
- Genesis motor control: https://genesis-world.readthedocs.io/en/v1.4.0/user_guide/getting_started/control_your_robot.html
- Genesis rendering: https://genesis-world.readthedocs.io/en/v1.4.0/user_guide/rendering/
- Stable-Baselines3 custom Gymnasium env: https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

## 7. OmniCampus提出用PDFの自動作成

`01_physics_test.bat` と `03_evaluate.bat` が成功したら **`07_generate_submission_pdf.bat`** を実行し、**氏名ではなくOmniCampusアカウントID**を入力する。実測値を取り込み、`outputs/説明資料_PAI最終課題_アカウントID.pdf` を自動生成する。データがない場合や物理テスト未達成の場合は生成を拒否する。

※ 評価エピソード数、失敗例、動画内容などについて人間が最終確認し、講義の提出条件と合うよう調整すること。レポート自動生成は結果を捏造しない。動画はYouTubeにアップロードし、GitHubはpublicに設定してOmniCampusに両URLを提出する。
