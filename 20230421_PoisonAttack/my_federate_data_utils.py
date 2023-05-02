# -*-coding:utf-8 -*-

"""
# File       : my_federate_data_utils.py
# Time       ：2023/4/26 22:33
# Author     ：
# Email      ：
# Description：
"""
from syft import FederatedDataLoader
from syft.frameworks.torch.fl.dataset import FederatedDataset, BaseDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from sklearn.utils import shuffle
import torch
from torch.utils.data import TensorDataset, DataLoader
import syft as sy

hook = sy.TorchHook(torch)


def get_federated_workers(client_cnt=10):
    workers = []
    for i in range(client_cnt):
        workers.append(sy.VirtualWorker(hook=hook, id=f'client_{i}'))
    return workers


def get_federated_dfs(all_client_df, serve_df, label_col='Attack_label', iid=True,  poison_rate=0, client_cnt=10,
                      poison_client_cnt=0):

    all_labels = list(serve_df[label_col].drop_duplicates())
    client_rate = 1/client_cnt
    federated_dfs = {'serve': serve_df}

    client_dfs = []
    if iid:
        for _ in range(client_cnt):
            client_df = all_client_df.sample(frac=2*client_rate)
            client_dfs.append(client_df)
    else:
        for _ in range(client_cnt):
            client_df = []
            for label in all_labels:
                cur_label_df = all_client_df.loc[all_client_df[label_col] == label, :]
                cur_label_client_df = cur_label_df.sample(frac=client_rate*(1+random.randint(0, 100)/100*2))
                client_df.append(cur_label_client_df)
            client_df = pd.concat(client_df)
            client_df = shuffle(client_df)
            client_dfs.append(client_df)

    # poison
    if poison_client_cnt:
        poison_dfs = []
        for client_df in client_dfs[:poison_client_cnt]:
            health_df, poison_df = train_test_split(client_df, test_size=poison_rate)
            poison_df[label_col] = poison_df[label_col].map(lambda x: random.choice(list(set(all_labels) - {x})))
            poison_df = pd.concat([health_df, poison_df])
            poison_df = shuffle(poison_df)
            poison_dfs.append(poison_df)
        federated_dfs['clients'] = poison_dfs + client_dfs[poison_client_cnt:]
    else:
        federated_dfs['clients'] = client_dfs

    return federated_dfs


def get_federated_dataset(dfs, workers, label_col='Attack_label'):
    datasets = []
    for df, worker in zip(dfs, workers):
        x = torch.tensor(df.drop([label_col], axis=1).values, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(df[label_col].values, dtype=torch.long)
        x = x.send(worker)
        y = y.send(worker)
        datasets.append(BaseDataset(x, y))
    return FederatedDataset(datasets)


def get_federated_dataloader(federated_dataset, batch_size=125, shuffle=True):
    return FederatedDataLoader(federated_dataset, batch_size=batch_size, shuffle=shuffle)


def get_serve_dataset(df, label_col='Attack_label'):
    x = torch.tensor(df.drop(['Attack_label', 'Attack_type'], axis=1).values, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(df[label_col].values, dtype=torch.long)
    return TensorDataset(x, y)


def get_serve_dataloader(dataset, batch_size=125):
    return DataLoader(dataset, batch_size, shuffle=False)
