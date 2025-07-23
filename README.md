🚀 PyTorchによる連合学習(FedAvg)シミュレーター
📝 概要 (Overview)
これは、PyTorchを用いて連合学習（Federated Learning, FL）の代表的なアルゴリズムである Federated Averaging (FedAvg) を実装したシミュレーターです。

プライバシーを保護しながら分散したデータで機械学習モデルを訓練する連合学習の挙動を、特にクライアント間のデータ分布が不均一なNon-IID（非独立同一分布）環境において検証することを目的としています。

✨ 主な特徴 (Features)
FedAvgアルゴリズム: 連合学習の基本的なアルゴリズムを忠実に実装。

Non-IIDデータ分割: ディリクレ分布 (Dirichlet distribution) を用いて、クライアント間のデータ分布の偏り（不均一性）を柔軟にシミュレート可能。

最適化手法の選択: Adam と SGD の2つの最適化手法をコマンドラインから簡単に切り替えて実験可能。

高い再現性: 乱数シードを完全に固定し、誰が実行しても同じ結果が得られる実験環境を構築。

柔軟な実験設定: argparseにより、クライアント数、学習率、Non-IIDの度合い (alpha) などのハイパーパラメータをコマンドラインから簡単に変更可能。

結果の自動保存: 各エポックのグローバルモデルの精度をpandasで集計し、CSVファイルとして自動で出力。

🛠️ 技術スタック (Tech Stack)
Core: Python 3.9

Machine Learning: PyTorch

Data Handling: NumPy, Pandas

Dataset: MNIST

📦 セットアップと実行方法 (Installation & Usage)
1. 準備
まず、リポジトリをクローンし、必要なライブラリをインストールします。
（your-repositoryの部分は、実際のリポジトリ名に書き換えてください）

Bash

# リポジトリをクローン
git clone https://github.com/Manstein0704/your-repository.git
cd your-repository

# 必要なライブラリをインストール
pip install torch torchvision pandas numpy
2. 実行
以下のコマンドで学習を開始します。--optimizer 引数で adam または sgd を指定できます。

Adamを使用する場合 (デフォルト):

Bash

python main.py --epochs 50 --num_clients 100 --selected_clients 10 --lr 0.001 --dirichlet_alpha 0.5 --optimizer adam
SGDを使用する場合:

Bash

python main.py --epochs 50 --num_clients 100 --selected_clients 10 --lr 0.01 --dirichlet_alpha 0.5 --optimizer sgd
主な引数:

--epochs: グローバルな学習ラウンド数

--num_clients: 総クライアント数

--selected_clients: 毎ラウンドで学習に参加するクライアント数

--dirichlet_alpha: データの不均一性を制御（値が小さいほどNon-IID性が強くなる）

--optimizer: adam または sgd を選択

--lr: 学習率

引数の詳細は --help で確認できます。

Bash

python main.py --help
学習が完了すると、実行時のパラメータに基づいたファイル名（例: FL_0.5_cohort_10_clients_100_Opt:adam_lr:0.001.csv）で結果が保存されます。

💡 工夫した点・設計思想 (Key Learnings & Design Choices)
このプロジェクトでは、単にアルゴリズムを実装するだけでなく、研究や開発に応用できるような、より実践的で信頼性の高いシミュレーターを目指しました。

現実的なシナリオの追求 (Non-IID環境):
実際の連合学習では、各ユーザー（クライアント）が持つデータは偏っているのが普通です。この偏り（Non-IID性）がモデルの学習にどう影響するかを検証するため、ディリクレ分布を採用しました。alphaパラメータを調整することで、全クライアントがほぼ同じデータを持つIIDな状況から、特定の数字しか持たない極端なNon-IIDまで、様々なシナリオを再現できます。

科学的検証のための再現性:
機械学習の実験において、再現性は極めて重要です。このシミュレーターでは、torch, numpy, random の全ての乱数シードを固定し、cuDNNの決定的動作を有効にすることで、誰がどの環境で実行しても同じ学習結果が得られるように設計しました。これにより、ハイパーパラメータ変更の影響だけを純粋に評価することが可能になります。

拡張性と柔軟性の確保:
argparseを全面的に採用することで、コードを直接変更することなく、様々な条件下での実験を簡単に行えるようにしました。これにより、「オプティマイザの違いによる収束性の比較」や「クライアント数が増えると収束速度はどうなるか？」といった問いに対する体系的な分析が容易になります。この設計は、将来的に新しいアルゴリズムやモデルを追加する際の拡張性も考慮しています。

🌱 今後の展望 (Future Work)
[ ] FedAvg以外の連合学習アルゴリズム（FedProx, SCAFFOLDなど）の実装

[ ] CNNなど、より複雑なモデルへの対応

[ ] CIFAR-10など、MNIST以外のデータセットへの対応

[ ] クライアントの通信コストや計算コストのシミュレーション機能

[ ] 実験結果を自動で可視化（グラフ描画）する機能の追加
