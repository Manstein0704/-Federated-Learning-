# 🚀 PyTorchによる連合学習(FedAvg)シミュレーション


[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 概要 (Overview)
これは、**PyTorch**を用いて連合学習（Federated Learning, FL）の代表的なアルゴリズムである **Federated Averaging (FedAvg)** を実装したシミュレーターです。

プライバシーを保護しながら分散したデータで機械学習モデルを訓練する連合学習の挙動を、特にクライアント間のデータ分布が不均一な**Non-IID（非独立同一分布）環境**において検証することを目的としています。

---

## ✨ 主な特徴 (Features)
* **FedAvgアルゴリズム:** 連合学習の基本的なアルゴリズムを忠実に実装。
* **Non-IIDデータ分割:** ディリクレ分布 (`Dirichlet distribution`) を用いて、クライアント間のデータ分布の偏り（不均一性）を柔軟にシミュレート可能。
* **最適化手法の選択:** `Adam` と `SGD` の2つの最適化手法をコマンドラインから簡単に切り替えて実験可能。
* **高い再現性:** 乱数シードを完全に固定し、誰が実行しても同じ結果が得られる実験環境を構築。
* **柔軟な実験設定:** `argparse`により、クライアント数、学習率、Non-IIDの度合い (`alpha`) などのハイパーパラメータをコマンドラインから簡単に変更可能。
* **結果の自動保存:** 各エポックのグローバルモデルの精度を`pandas`で集計し、CSVファイルとして自動で出力。

---

## 🛠️ 技術スタック (Tech Stack)
* **Core:** Python 3.9+
* **Machine Learning:** PyTorch 2.0+
* **Data Handling:** NumPy, Pandas
* **Dataset:** MNIST (from torchvision)

---

## 📦 セットアップと実行方法 (Installation & Usage)
```bash
### 1. 準備

まず、リポジトリをクローンし、必要なライブラリをインストールします。


# リポジトリをクローン
git clone https://github.com/Manstein0704/your-repository.git
cd your-repository

# 必要なライブラリをインストール
pip install -r requirements.txt

2. 実行例
python main.py \
    --num_clients 100 \
    --selected_clients 10 \
    --epochs 50 \
    --local_epochs 1 \
    --dirichlet_alpha 0.5 \
    --optimizer adam \
    --lr 0.001

⚙️ コマンドライン引数一覧 (Arguments)
引数	デフォルト値	説明
--batch_size	64	テスト用のバッチサイズ
--train_batch_size	64	各クライアントでのトレーニング時のバッチサイズ
--num_clients	128	クライアントの総数
--selected_clients	16	各エポックで選択されるクライアントの数
--epochs	100	エポック数（通信ラウンド数）
--local_epochs	1	各クライアントでのローカルトレーニングの回数
--dirichlet_alpha	1e21	ディリクレ分布のαパラメータ（小さいほどNon-IIDが強くなる）
--optimizer	"adam"	adam または sgd
--lr	0.0001	学習率
--momentum	0	SGD使用時のモメンタム係数

📊 出力結果
実行が完了すると、各エポックでのテスト精度を記録したCSVファイルが自動生成されます。

例:./FL_0.5_cohort_10_clients_100_Opt:adam_lr:0.001.csv
ファイルには以下が含まれます：
各イテレーションにおけるエポックごとの精度
エポックごとの平均精度


📈 実験例と可視化
このシミュレーターを使えば、以下のような実験が可能です：
データの非IID性（α）の変化による精度への影響
クライアント数・選択数の調整による性能変化
最適化手法や学習率による収束速度の違い
matplotlibなどを用いたCSVファイルの可視化も簡単に行えます。


🔍 今後の拡張アイデア (Future Work)
異なるモデル構造（CNNなど）への対応
クライアント側での早期停止の実装
通信オーバーヘッドの計測
重み付き平均以外の集約手法（例: Krum, Trimmed Mean）
<img width="1120" height="700" alt="federated_accuracy" src="https://github.com/user-attachments/assets/707550d3-4b01-4dba-bdc9-c177d8c21428" />

