# -*-coding:utf-8 -*-

"""
# File       : main.py
# Time       ：2023/4/21 22:43
# Author     ：Linhan XU
# Email      ：xulinhan1233@gmail.com
# Description：
"""
import copy
import json
import sys

import yaml
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import torch
from tqdm import tqdm, trange
from torch import optim
import pandas as pd
from my_federate_data_utils import get_federated_workers, get_federated_dfs, get_federated_dataset, \
    get_federated_dataloader, get_serve_dataset, get_serve_dataloader
from model import CnnNet

Attack_Type_to_name = {0: 'Normal',
 1: 'Password',
 2: 'Vulnerability_scanner',
 3: 'DDoS_TCP',
 4: 'DDoS_UDP',
 5: 'DDoS_ICMP',
 6: 'DDoS_HTTP',
 7: 'Uploading',
 8: 'XSS',
 9: 'SQL_injection',
 10: 'Backdoor',
 11: 'Port_Scanning',
 12: 'Ransomware',
 13: 'Fingerprinting',
 14: 'MITM'}

Attack_Label_to_name = {
    0: 'Normal',
    1: 'Attack'
}


def get_federated_models(model, workers, device):
    federated_models = dict()
    for worker in workers:
        client_model = copy.deepcopy(model)
        client_model = client_model.to(device)
        client_model.send(worker)
        federated_models[worker] = client_model
    return federated_models


def get_federated_opts(federated_models, opt='SGD', opt_params={'lr': .01}):
    federated_opts = dict()
    for worker, model in federated_models.items():
        federated_opts[worker] = getattr(optim, opt)(model.parameters(), **opt_params)
    return federated_opts


def federated_train(federated_dataloader, workers, model, opt='SGD', opt_params={'lr': .0001}, device='cpu',
                      epoch=3):
    federated_models = get_federated_models(model, workers, device=device)
    federated_opts = get_federated_opts(federated_models, opt, opt_params)
    for ep in range(epoch):
        for worker, client_model in federated_models.items():
            client_model.train()
        iter_dataloader = iter(federated_dataloader)
        losses = []
        with trange(len(iter_dataloader)) as t:
            for _ in t:
                x, y = iter_dataloader.__next__()
                x = x.to(device)
                y = y.to(device)
                location = x.location
                client_model = federated_models[location]
                opt = federated_opts[location]
                opt.zero_grad()
                pred = client_model(x)
                loss = client_model.loss(pred, y)
                loss.backward()
                opt.step()
                losses.append(loss.get())
                t.set_description(f'Train ClientsEpoch:{ep} {location.id}')
                t.set_postfix(loss=(sum(losses)/len(losses)).data.cpu().numpy())

    # model merge
    local_models = []
    for _, client_model in federated_models.items():
        local_models.append(client_model.get())
    model_cnt = len(local_models)
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            avg_param = sum([list(m.parameters())[idx].data for m in local_models])/model_cnt
            param.set_(avg_param)

    return model


def evaluate(model, test_dataloader, object):
    iter_dataloader = iter(test_dataloader)
    losses = []
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        with trange(len(iter_dataloader)) as t:
            for _ in t:
                x, y = iter_dataloader.__next__()
                loss = model.loss(model(x), y)
                losses.append(loss)
                preds.append(model.predict(x))
                targets.append(y)
                t.set_description('Eval')
                t.set_postfix(loss=(sum(losses)/len(losses)).data.cpu().numpy())
    preds = torch.cat(preds).reshape((-1)).cpu().numpy()
    targets = torch.cat(targets).reshape((-1)).cpu().numpy()

    # loss
    loss = (sum(losses)/len(losses)).data.cpu().numpy()

    # metric report
    if object == 'binary':
        labels = Attack_Label_to_name
    else:
        labels = Attack_Type_to_name
    f1 = f1_score(targets, preds, labels=list(labels.keys()), average=None)
    precision = precision_score(targets, preds, labels=list(labels.keys()), average=None)
    recall = recall_score(targets, preds, labels=list(labels.keys()), average=None)
    acc = accuracy_score(targets, preds)
    round_metric_report = {labels[label]: {'f1': f, 'precision': p, 'recall': r} for label, f, p, r in zip(list(labels.keys()), f1, precision, recall)}
    round_metric_report['all'] = {'acc': acc}
    round_metric_report['loss'] = loss

    # print
    to_print = json.dumps(round_metric_report, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':'))
    print(to_print)

    return round_metric_report, loss


if __name__ == '__main__':
    conf_path = sys.argv[1]
    mode = sys.argv[2]

    with open(conf_path) as f:
        conf = f.read()
    conf = yaml.safe_load(conf)
    print('----------------------------conf-----------------------------')
    print(json.dumps(conf, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':')))
    print('----------------------------conf-----------------------------')

    workers = get_federated_workers(conf['data']['client_cnt'])

    # data init
    if conf['object'] == 'binary':
        all_client_df = pd.read_csv('data/preprocessed_DNN_train_balance_binary.csv', low_memory=False)
    else:
        all_client_df = pd.read_csv('data/preprocessed_DNN_train_balance_multi.csv', low_memory=False)
    serve_df = pd.read_csv('data/preprocessed_DNN_test.csv', low_memory=False)
    if mode == 'debug':
        all_client_df = all_client_df.sample(1000)
        serve_df = serve_df.sample(100)

    federated_dfs = get_federated_dfs(all_client_df, serve_df, iid=conf['data']['iid'],
                                      poison_rate=conf['data']['poison_rate'],
                                      label_col=conf['data']['label_col'],
                                      client_cnt=conf['data']['client_cnt'],
                                      poison_client_cnt=conf['data']['poison_client_cnt']
                                      )

    client_dfs = federated_dfs['clients']
    federated_dataset = get_federated_dataset(dfs=client_dfs, workers=workers, label_col=conf['data']['label_col'])
    federated_dataloader = get_federated_dataloader(federated_dataset, batch_size=conf['data']['batch_size'])

    serve_df = federated_dfs['serve']
    serve_dataset = get_serve_dataset(serve_df, conf['data']['label_col'])
    serve_dataloader = get_serve_dataloader(dataset=serve_dataset)

    # model init
    local_model = CnnNet(conf['object'], conf['train']['drop_out'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # loss_func = BCEWithLogitsLoss()

    cur_loss = float('inf')
    un_improve_round = 0
    metric_report = []
    for epoch in range(conf['train']['global_epoch']):
        print(f'------------------Global Epoch {epoch}------------------')
        local_model = federated_train(federated_dataloader, workers, local_model, opt=conf['train']['opt'],
                                      opt_params=conf['train']['opt_params'],
                                      device=device,
                                      epoch=conf['train']['client_epoch'])
        round_metric_report, loss = evaluate(model=local_model, test_dataloader=serve_dataloader,
                                              object=conf['object'])

        # save metric report
        round_metric_report['round'] = epoch
        metric_report.append(round_metric_report)
        if loss < cur_loss:
            cur_loss = loss
            torch.save(local_model.state_dict(), conf['path']['model_save_path'])
            un_improve_round = 0
        else:
            un_improve_round += 1
        if un_improve_round >= 1:  # lr down
            conf['train']['opt_params']['lr'] /= 10
        if un_improve_round >= 4:  # early_stop
            break
    with open(conf['path']['report_save_path'], 'w', encoding='utf-8') as f:
        to_write = json.dumps(metric_report)
        f.write(to_write)
