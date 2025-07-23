# 🚀 PyTorchによる連合学習(FedAvg)シミュレーター

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 概要 (Overview)

これは、**PyTorch**を用いて連合学習（Federated Learning, FL）の代表的なアルゴリズムである **Federated Averaging (FedAvg)** を実装したシミュレーターです。

プライバシーを保護しながら分散したデータで機械学習モデルを訓練する連合学習の挙動を、特にクライアント間のデータ分布が不均一な**Non-IID（非独立同一分布）環境**において検証することを目的としています。

<br>
## ✨ 主な特徴 (Features)

* **FedAvgアルゴリズム:** 連合学習の基本的なアルゴリズムを忠実に実装。
* **Non-IIDデータ分割:** ディリクレ分布 (`Dirichlet distribution`) を用いて、クライアント間のデータ分布の偏り（不均一性）を柔軟にシミュレート可能。
* **高い再現性:** 乱数シードを完全に固定し、誰が実行しても同じ結果が得られる実験環境を構築。
* **柔軟な実験設定:** `argparse`により、クライアント数、学習率、Non-IIDの度合い (`alpha`) などのハイパーパラメータをコマンドラインから簡単に変更可能。
* **結果の自動保存:** 各エポックのグローバルモデルの精度を`pandas`で集計し、CSVファイルとして自動で出力。

---

## 🛠️ 技術スタック (Tech Stack)

* **Core:** Python 3.9
* **Machine Learning:** PyTorch
* **Data Handling:** NumPy, Pandas
* **Dataset:** MNIST

---

## 📦 セットアップと実行方法 (Installation & Usage)

### 1. 準備

まず、リポジトリをクローンし、必要なライブラリをインストールします。

```bash
# リポジトリをクローン
git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
cd your-repository

# 必要なライブラリをインストール
pip install -r requirements.txt
# (もしrequirements.txtがなければ)
# pip install torch torchvision pandas numpy
