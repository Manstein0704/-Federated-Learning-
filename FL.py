import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import copy
import time
import random
 
# モデル定義
class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)
 
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
 
# プログラム内でシードリストを定義
# [1, 11, 21, 31, 41, 52, 61, 71, 81, 91, 3, 13, 23, 33, 43, 54, 63, 73, 83, 93]
seed_list = [11]
 
# Dirichlet分割関数
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
 
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        splits = (np.cumsum(fracs)[:-1] * len(c)).astype(int)
        for i, idcs in enumerate(np.split(c, splits)):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs
 
# メイン関数
def main(args):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(f"Total epochs: {args.epochs}, Local epochs per client: {args.local_epochs}, "
          f"Number of clients: {args.num_clients}, Dirichlet alpha: {args.dirichlet_alpha}, "
          f"selected_clients: {args.selected_clients}, optimizer: {args.optimizer}, lr: {args.lr}")
 
    # データセットのロード
    trans_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('data/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('data/', train=False, download=True, transform=trans_mnist)
    test_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
 
    time_start = time.time()
    print("------------------------Federated Learning-------------------------------")
 
    epoch_accuracies = [[] for _ in range(args.epochs)]
 
    for iteration, seed in enumerate(seed_list, 1):  # イテレーションごとのシード固定
        print(f"Iteration {iteration} with Seed {seed}")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.benchmark = False
 
 
        train_labels = np.array(dataset_train.targets)
        client_idcs = dirichlet_split_noniid(train_labels, alpha=args.dirichlet_alpha, n_clients=args.num_clients)
 
        client_datasets, client_dataloaders = [], []
        for client in range(args.num_clients):
            idx = list(client_idcs[client])
            if not len(idx):
                print(f"Warning: Client {client} has no samples.")
                client_datasets.append(None)
                client_dataloaders.append(None)
            else:
                sub_dataset = Subset(dataset_train, idx)
                dl = DataLoader(sub_dataset, batch_size=args.train_batch_size, shuffle=True)
                client_datasets.append(sub_dataset)
                client_dataloaders.append(dl)
 
        global_model = MLP2().to(device)
        criterion = nn.CrossEntropyLoss()
 
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            collected = []
            selected = random.sample(range(args.num_clients), args.selected_clients)
            print(f"Selected Clients: {selected}")
 
            for cid in selected:
                dl = client_dataloaders[cid]
                if dl is None:
                    print(f"Skipping Client {cid} due to no data.")
                    continue
 
                local_model = copy.deepcopy(global_model)
                # オプティマイザー選択
                if args.optimizer.lower() == 'adam':
                    optimizer = optim.Adam(local_model.parameters(), lr=args.lr)
                else:
                    optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum)
 
                local_samples = 0
                local_model.train()
 
                for _ in range(args.local_epochs):
                    for x, y in dl:
                        x, y = x.to(device), y.to(device)
                        local_samples += x.size(0)
                        optimizer.zero_grad()
                        out = local_model(x)
                        loss = criterion(out, y)
                        loss.backward()
                        optimizer.step()
 
                collected.append((copy.deepcopy(local_model.state_dict()), local_samples))
 
            if not collected:
                print("No clients participated in this epoch.")
                continue
 
            # グローバルモデル集約
            gsd = global_model.state_dict()
            total = sum(n for _, n in collected)
            for k in gsd.keys():
                gsd[k] = torch.stack([w[k] * n / total for w, n in collected]).sum(dim=0)
            global_model.load_state_dict(gsd)
 
            # 評価
            global_model.eval()
            correct = total_test = 0
            with torch.no_grad():
                for x, y in test_dataloader:
                    x, y = x.to(device), y.to(device)
                    preds = global_model(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total_test += y.size(0)
            acc = correct / total_test * 100
            print(f"Global Model Accuracy: {acc:.2f}%")
            epoch_accuracies[epoch].append(acc)
 
    df = pd.DataFrame(epoch_accuracies,
                      index=[f"Epoch_{i+1}" for i in range(args.epochs)])
    df.columns = [f"Iteration_{i+1}" for i in range(len(epoch_accuracies[0]))]
    df["AverageAccuracy"] = df.mean(axis=1)
    out_path = f"./FL_{args.dirichlet_alpha}_cohort_{args.selected_clients}_clients_{args.num_clients}_Opt:{args.optimizer}_lr:{args.lr}.csv"
    df.to_csv(out_path, index=False)
 
    print(f"Total Execution Time: {time.time() - time_start:.4f} seconds")
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--num_clients', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--dirichlet_alpha', type=float, default=10e20)
    parser.add_argument('--selected_clients', type=int, default=16)
    parser.add_argument('--optimizer',type=str,default='adam',choices=['adam', 'sgd'])
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--momentum',type=float,default=0)
    args = parser.parse_args()
    main(args)
