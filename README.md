# OlfaLink Scout
### Reinforcement Learning for Autonomous Odor Source Localization using Genesis

Genesis WorldとPPO（Proximal Policy Optimization）を用いた、**におい発生源の自律探索ロボットシミュレーション**です。

本プロジェクトでは、差動二輪ロボットが局所的なにおい情報と周囲の障害物情報を観測し、強化学習によって発生源へ移動する行動方策の獲得を目指しました。

ロボットの運動・接触はGenesisの物理エンジンで計算し、におい分布は数理モデルで表現しています。

## 1. Background

室内環境において、生活臭や異臭を検知できたとしても、その発生源の位置まで特定できるとは限りません。

においは空間的に分布するため、特定の位置に設置されたセンサーの情報だけでは、発生源の位置を判断することが難しい場合があります。

そこで本プロジェクトでは、ロボットが環境内を移動しながらにおい情報を取得し、発生源を探索するシステムを開発しました。

特に、あらかじめ発生源までの経路を与えるのではなく、**強化学習によって観測情報から移動行動を決定する**ことを目標としています。

将来的には、においの検知・可視化を目的としたOlfaLinkと組み合わせ、発生源の調査を支援するシステムへの発展を想定しています。

## 2. Objective

本プロジェクトの目的は、以下の3点です。

1. Genesisを用いて、差動二輪ロボットが物理法則に従って移動するシミュレーション環境を構築する。
2. 局所的なにおい観測と障害物情報を用いて、PPOによる自律探索方策を学習する。
3. ランダム探索との比較実験を通じて、学習した方策の性能と課題を評価する。

## 3. System Architecture

システムは以下の要素で構成されます。

| Component | Technology | Description |
|---|---|---|
| Physics Simulation | Genesis World | ロボットの運動・接触・関節制御 |
| Robot Model | URDF | 差動二輪ロボット |
| Environment | Gymnasium | 状態・行動・報酬の定義 |
| Reinforcement Learning | Stable-Baselines3 / PPO | 行動方策の学習 |
| Deep Learning | PyTorch | 方策ネットワークの計算 |
| Odor Simulation | NumPy | 仮想におい分布・センサーノイズ |
| Visualization | Genesis / Matplotlib | 3D動画・移動軌跡の出力 |

### Processing Flow

```text
Genesis Simulation Environment
             |
             v
      Sensor Observation
       - Robot Position
       - Robot Orientation
       - Odor Signal
       - Distance Sensors
             |
             v
        PPO Policy
             |
             v
       Action Selection
             |
             v
      Wheel Motor Control
             |
             v
       Genesis Physics
             |
             v
       Next Observation
```

ロボットは観測と行動を繰り返し、発生源への到達を目指します。

## 4. Simulation Environment

### 4.1 Robot

URDFで定義した差動二輪ロボットを使用します。

左右の車輪に角速度指令を与えることで、前進・後退・旋回を実現します。

ロボットの移動は座標を直接変更する方式ではなく、Genesisの剛体物理シミュレーションに基づいています。

### 4.2 Environment

シミュレーション環境には、以下を配置します。

- 床および壁
- 障害物
- 差動二輪ロボット
- 仮想的なにおい発生源

発生源の位置は複数の候補から設定されます。

3D映像に表示される発生源マーカーは可視化用であり、その座標が直接PPOの観測情報として与えられることはありません。

### 4.3 Odor Distribution

におい分布は、方向によって広がりが異なる2次元ガウス関数によって近似します。

\[
C(x,y)=\exp\left[-\frac{1}{2}
\left(
\frac{x'^2}{\sigma_x^2}
+
\frac{y'^2}{\sigma_y^2}
\right)\right]
\]

ここで、\(x',y'\) は発生源を中心として回転変換した座標です。

風向に相当するパラメータによって濃度分布の向きを設定しています。また、センサー観測にはノイズと一次遅れを導入しています。

**本モデルは実際のガス拡散を再現したCFDシミュレーションではありません。** におい信号は無次元の仮想値であり、実際のVOC濃度やppmを表していません。

障害物の距離情報についても、5方向の解析的なレイキャストを使用しており、Genesisの実LiDARセンサーは使用していません。

## 5. Reinforcement Learning

### 5.1 Algorithm

強化学習アルゴリズムにはPPO（Proximal Policy Optimization）を採用しました。

PPOは方策勾配法に基づくActor-Critic系のアルゴリズムであり、方策の更新幅を制限することで学習の安定化を図ります。

実装にはStable-Baselines3を使用しています。

### 5.2 Observation Space

観測空間は15次元の連続値ベクトルです。

主な情報は以下のとおりです。

- ロボットの位置・向き
- 現在のにおい強度
- におい強度の時間的変化
- 5方向の障害物距離
- 直前の行動
- 経過ステップ
- ロボットの移動速度・旋回速度

発生源の正解座標は、方策ネットワークの観測ベクトルに含めていません。

### 5.3 Action Space

行動空間は5種類の離散行動です。

| Action ID | Action |
|---|---|
| 0 | Stop |
| 1 | Forward |
| 2 | Turn Left |
| 3 | Turn Right |
| 4 | Reverse |

選択された行動は左右車輪の角速度指令に変換され、Genesis上で実行されます。

### 5.4 Reward Design

報酬は以下の要素を組み合わせて設計しています。

- 発生源への接近に対する報酬
- におい強度の増加に対する報酬
- 発生源への到達に対する成功報酬
- ステップごとのペナルティ
- 障害物付近での停滞や領域外への移動に対するペナルティ

発生源からの真の距離は、シミュレーション環境が計算する学習報酬と評価に使用します。

なお、この報酬設計はシミュレーション上の正解情報を利用するものであり、実機で同じ報酬を直接計算できるとは限りません。

### 5.5 Success Condition

ロボットと発生源の距離が **0.32 m以下**になった場合を成功と判定します。

1エピソードの最大行動ステップ数は160です。

## 6. Experimental Results

### 6.1 Execution Environment

- OS: Windows 11
- Python: 3.11.9 (64-bit)
- Genesis World: 1.4.0
- PyTorch: CPU版
- Reinforcement Learning: PPO
- Training Steps: 20,480
- Evaluation Episodes: 20 per policy

### 6.2 Physics Simulation Verification

Genesisの物理エンジンによる前進・旋回動作を検証しました。

| Metric | Result |
|---|---:|
| Forward Progress | 0.219 m |
| Yaw Rotation | 1.082 rad |
| Observation Validity | Passed |

車輪の速度制御によってロボットが移動・旋回することを確認しました。

### 6.3 PPO Training

PPOによる学習を20,480ステップ実行し、学習済みモデルを保存しました。

- Training Steps: 20,480
- Training Time: 約60分42秒
- Execution Device: CPU

### 6.4 Evaluation Results

学習済みPPOとランダム探索を、それぞれ20エピソードで比較しました。

| Metric | Random Policy | PPO |
|---|---:|---:|
| Episodes | 20 | 20 |
| Successful Episodes | 0 | 1 |
| Success Rate | 0% | 5% |
| Mean Episode Length | 160.00 | 155.45 |
| Mean Final Distance | 2.713 m | 1.865 m |

PPOはランダム探索と比較して、発生源までの最終平均距離が短い結果となりました。

一方で、発生源の発見成功率は5%にとどまっており、現段階では安定した探索方策を獲得できていません。

また、評価試行数は20回と少ないため、今回の結果だけでPPOの優位性を統計的に結論づけることはできません。

### 6.5 3D Simulation Demo

学習済みPPOモデルを使用し、Genesis上で3Dシミュレーション動画を生成しました。

録画した64秒のシミュレーションでは、以下の結果が得られました。

| Episode | Result | Steps |
|---|---|---:|
| 0 | Success | 85 |
| 1 | Not Found | 160 |
| 2 | Not Found within recording | 155 |

Episode 0では、学習済みPPOによってロボットが発生源に到達しました。

Episode 2は録画時間の終了によって途中で切り上げられているため、最大160ステップまでの完全な評価エピソードではありません。

この動画は探索成功例を示していますが、モデル全体の成功率を示すものではありません。性能評価には前述の20エピソードの比較結果を使用しています。

## 7. Discussion

本実験では、Genesisによる物理シミュレーションとPPOを統合し、におい発生源を探索するロボットシステムを実装できました。

さらに、学習済みPPOによって実際に発生源へ到達したエピソードが確認されました。

しかし、評価成功率は5%であり、探索性能には大きな改善の余地があります。

考えられる要因として、以下が挙げられます。

1. 学習ステップ数が十分でない可能性
2. 局所的なにおい観測だけでは探索に必要な情報が不足する可能性
3. 報酬関数が効率的な探索行動を十分に誘導できていない可能性
4. 発生源位置が異なる環境に対する汎化性能が不十分である可能性

これらは現時点では仮説であり、追加実験による検証が必要です。

また、本研究では仮想のにおい分布を用いているため、実環境における気流の乱れ、ガス拡散、センサー応答などを十分に再現できていません。

## 8. Installation

### Requirements

- Windows 10 / 11 (64-bit)
- Python 3.11 (64-bit)
- Git
- 初回インストール時のインターネット接続

### Clone Repository

```powershell
git clone https://github.com/mathkumamoto2/OlfaLink-Scout.git
cd OlfaLink-Scout
```

### Setup

```powershell
.\setup_windows.bat
```

Python仮想環境と必要なライブラリをインストールします。

### Physics Test

```powershell
.\01_physics_test.bat
```

ロボットの前進・旋回動作を検証します。

### Train PPO

```powershell
.\02_train_ppo.bat
```

PPOによる学習を実行します。

学習済みモデルは以下に保存されます。

```text
outputs/ppo_genesis.zip
```

### Evaluate

```powershell
.\03_evaluate.bat
```

ランダム探索と学習済みPPOの比較評価を実行します。

### Record 3D Simulation

```powershell
.\04_record_3d_ppo.bat
```

学習済みPPOを使用して3D動画を生成します。

```text
outputs/genesis_3d_demo.mp4
```

なお、GenesisのライブビューワはGPUドライバやOpenGL環境によって動作しない場合があります。本実験環境ではライブビューワの初期化エラーが発生しましたが、3D動画の録画には成功しました。

## 9. Project Structure

```text
OlfaLink-Scout/
│
├── scout_genesis/
│   ├── core.py
│   ├── env.py
│   ├── cli.py
│   ├── viz.py
│   └── report.py
│
├── assets/
│   └── scout_diffdrive.urdf
│
├── tests/
│
├── setup_windows.bat
├── 01_physics_test.bat
├── 02_train_ppo.bat
├── 03_evaluate.bat
├── 04_record_3d_ppo.bat
├── 05_trajectory_only.bat
├── 06_preview_3d_scripted.bat
├── 07_generate_submission_pdf.bat
│
├── requirements.txt
└── README.md
```

学習済みモデル、評価結果、動画などは `outputs/` 以下に生成されます。

これらの生成ファイルはGitの管理対象から除外しているため、学習済みモデルを再現する場合は学習コマンドを実行してください。

## 10. Future Work

今後は以下の改良を検討しています。

**1. Reinforcement Learning**

- 報酬関数の改善
- 学習ステップ数の増加
- 探索履歴を考慮した方策
- より多様な初期位置・発生源位置での学習

**2. Odor Simulation**

- 時間変動する濃度分布
- 気流・障害物の影響
- センサー応答遅延の検証
- 実測データとの比較

**3. Physical AI**

- 複数センサーの情報統合
- 能動的知覚による探索効率の改善
- 実際のガスセンサーとの連携
- 実機ロボットへの展開

**4. OlfaLink Integration**

将来的にはOlfaLinkと連携し、においの検知・可視化に加えて、発生源の位置推定や探索結果を利用者に提示する機能への発展を目指します。

## 11. References

- [Genesis World Documentation](https://genesis-world.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Proximal Policy Optimization Algorithms — Schulman et al., 2017](https://arxiv.org/abs/1707.06347)

---

**OlfaLink Scout — Autonomous Odor Source Localization with Reinforcement Learning**

本プロジェクトは、Physical AI応用1講座の最終課題として開発しました。
