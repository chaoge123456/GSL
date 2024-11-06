import os
import copy
import time
import torch
import random
import argparse
import numpy as np
from math import floor
from pathlib import Path
from random import sample
from update import Byzantine
from server import Backdoor_train
from utils import load_predict_data
from collections import OrderedDict
from network import client_resnet, server_tasker
from train import CleanGroup, MaliciousGroup, predict
from torch.utils.tensorboard import SummaryWriter
from dataset.utils.generate_mnist import generate_mnist
from dataset.utils.generate_cifar10 import generate_cifar10
from dataset.utils.generate_fmnist import generate_fmnist
from dataset.utils.generate_cifar100 import generate_cifar100


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def partition(num_client, num_group):
    result = []
    client_list = [i for i in range(num_client)]
    random.shuffle(client_list)
    flo = floor(num_client / num_group)
    res = client_list[num_group*flo:]
    for i in range(num_group):
        cli = client_list[i*flo:(i+1)*flo]
        if i < len(res):
            cli.append(res[i])
        result.append(cli)
    return result


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def flatten_weights(weights):
    flattened_weights = []

    for weight in weights:
        flattened_weight = []
        for name in weight.keys():
            flattened_weight = (
                weight[name].view(-1)
                if not len(flattened_weight)
                else torch.cat((flattened_weight, weight[name].view(-1)))
            )

        flattened_weights = (
            flattened_weight[None, :]
            if not len(flattened_weights)
            else torch.cat((flattened_weights, flattened_weight[None, :]), 0)
        )
    return flattened_weights


def trimmed_mean(weights_attacked, num_attackers):
    flattened_weights = flatten_weights(weights_attacked)
    n, d = flattened_weights.shape
    median_weights = torch.median(flattened_weights, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(flattened_weights - median_weights), dim=0)
    sorted_weights = flattened_weights[sort_idx, torch.arange(d)[None, :]]

    mean_weights = torch.mean(sorted_weights[: n - 2 * num_attackers], dim=0)

    start_index = 0
    trimmed_mean_update = OrderedDict()
    for name, weight_value in weights_attacked[0].items():
        trimmed_mean_update[name] = mean_weights[
            start_index : start_index + len(weight_value.view(-1))
        ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))

    return trimmed_mean_update


def aggregation_average(groups, malicious_group_id):
    sm = []
    cm = []
    for g in groups:
        sm.append(copy.deepcopy(g.server.server_model.state_dict()))
        cm.append(copy.deepcopy(g.client.client_model.state_dict()))
    global_server = average_weights(sm)
    global_client = average_weights(cm)
    for g in groups:
        if g.id not in malicious_group_id:
            g.server.server_model.load_state_dict(global_server)
        # g.server.server_model.load_state_dict(global_server)
        g.client.client_model.load_state_dict(global_client)
    return global_client, global_server


def aggregation_multi_krum(groups, malicious_group_id, selection_set):
    sm = []
    cm = []
    for g in groups:
        if g.id in selection_set:
            sm.append(copy.deepcopy(g.server.server_model.state_dict()))
            cm.append(copy.deepcopy(g.client.client_model.state_dict()))
    global_server = average_weights(sm)
    global_client = average_weights(cm)
    for g in groups:
        if g.id not in malicious_group_id:
            g.server.server_model.load_state_dict(global_server)
        # g.server.server_model.load_state_dict(global_server)
        g.client.client_model.load_state_dict(global_client)
    return global_client, global_server


def aggregation_bulyan(groups, malicious_group_id, selection_set):
    sm = []
    cm = []
    for g in groups:
        if g.id in selection_set:
            sm.append(copy.deepcopy(g.server.server_model.state_dict()))
            cm.append(copy.deepcopy(g.client.client_model.state_dict()))
    global_server = trimmed_mean(sm, len(malicious_group_id))
    global_client = trimmed_mean(cm, len(malicious_group_id))
    for g in groups:
        if g.id not in malicious_group_id:
            g.server.server_model.load_state_dict(global_server)
        # g.server.server_model.load_state_dict(global_server)
        g.client.client_model.load_state_dict(global_client)
    return global_client, global_server


def global_predict(ep, args, logger, in_dim, global_client, global_server, cleantest, attacktest):
    clientmodel = client_resnet(args.level, in_dim)
    clientmodel.load_state_dict(global_client)
    clientmodel.to(args.device)
    servermodel = server_tasker(args.level)
    servermodel.load_state_dict(global_server)
    servermodel.to(args.device)
    accclean = predict(cleantest, clientmodel, servermodel, args.device)
    accattack = predict(attacktest, clientmodel, servermodel, args.device, clean=False)
    logger.add_scalar("Global/client", accclean, ep)
    logger.add_scalar("Global/client backdoor", accattack, ep)
    print("Global Client Accuracy:{:.4f}\tGlobal Client Attack:{:.4f}".format(accclean, accattack))


def args_parser(log_time):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument("--niid", type=bool, default=False, help='Non-iid or iid')
    parser.add_argument("--balance", type=bool, default=True, help='not balance or balance')
    parser.add_argument("--partition", type=str, default='dir', help='pat or dir')
    parser.add_argument("--type", type=str, default='ResNet18', help='type of network')

    parser.add_argument("--mode", type=str, default='defense', help='clean, attack, defense')
    parser.add_argument("--step", type=str, default='embed', help='fsha, plus, embed')
    parser.add_argument("--aggregation", type=str, default='krum', help='average, krum, bulyan')
    parser.add_argument('--select', type=int, default=3, help="number of select groups for update")

    parser.add_argument('--dataset', type=str, default='fmnist', help="name of dataset")
    parser.add_argument('--num_group', type=int, default=10, help="number of groups")
    parser.add_argument('--num_client', type=int, default=100, help="number of clients")
    parser.add_argument('--malicious', type=int, default=3, help="number of malicious server")
    parser.add_argument('--num_backdoor', type=int, default=250, help="number of backdoor sample")
    parser.add_argument('--server_sample', type=int, default=20000, help="number of server sample")
    parser.add_argument('--target', type=int, default=1, help="backdoor target label")
    parser.add_argument('--level', type=int, default=1, help="split network level")
    parser.add_argument('--weight_m', type=float, default=1.0, help="client loss weight")
    parser.add_argument('--weight_g', type=float, default=0.2, help="client loss weight")

    parser.add_argument('--batch_size', type=int, default=250, help='batch size of client training')
    parser.add_argument('--epochs', type=int, default=30, help="epochs of training")
    parser.add_argument("--device", type=str, default='cuda', help='cuda of cpu')

    parser.add_argument('--lr_client', type=float, default=0.0002, help='learning rate client')
    parser.add_argument('--lr_server', type=float, default=0.0002, help="learning rate server")
    parser.add_argument('--lr_D', type=float, default=0.0005, help="learning rate D")
    parser.add_argument('--gp', type=float, default=500., help="gradient penalty")
    parser.add_argument('--log_frequency', type=int, default=100, help="log_frequency")
    args = parser.parse_args()

    argsDict = args.__dict__
    path = './parser/' + log_time + '.txt'
    with open(path, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    return args


if __name__ == "__main__":

    log = str(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
    args = args_parser(log)
    seed_torch(args.seed)
    # logger = SummaryWriter(Path('./recoder/'+args.mode) / log)
    logger = SummaryWriter(Path('./recoder/test/') / log)


    if args.dataset == "mnist":
        in_dim = 1
        generate_mnist(args.num_client, args.niid, args.balance, args.partition)

    if args.dataset == "fmnist":
        in_dim = 1
        generate_fmnist(args.num_client, args.niid, args.balance, args.partition)

    if args.dataset == "cifar10":
        in_dim = 3
        generate_cifar10(args.num_client, args.niid, args.balance, args.partition)

    if args.dataset == "cifar100":
        in_dim = 3
        generate_cifar100(args.num_client, args.niid, args.balance, args.partition)

    testdata, _ = load_predict_data(args.dataset)
    cleantest = torch.utils.data.DataLoader(testdata, batch_size=2048, shuffle=True, num_workers=0, pin_memory=True)
    backdoor_data = copy.deepcopy(testdata)
    index_backdoor = [i for i in range(len(backdoor_data))]
    attacktest = torch.utils.data.DataLoader(Backdoor_train(backdoor_data, args.target, index_backdoor, in_dim), 2048, shuffle=True, num_workers=0, pin_memory=True)

    if args.mode == "clean":  # 正常训练
        id = 99
        member = [i for i in range(args.num_client)]
        group = CleanGroup(id, args, in_dim, args.device, member)
        for ep in range(args.epochs):
            print("\n******************************************Epoch:{}************************************************".format(ep))
            group.train(logger, cleantest)

    elif args.mode == "attack":  # 后门攻击
        id = 100
        member = [i for i in range(args.num_client)]
        group = MaliciousGroup(id, args, in_dim, args.device, member)
        for ep in range(args.epochs):
            print("\n******************************************Epoch:{}************************************************".format(ep))
            group.train(logger, cleantest, attacktest)

    else:  # 防御机制
        group_list = [i for i in range(args.num_group)]
        member_list = partition(args.num_client, args.num_group)
        if args.malicious == 0:
            malicious_group_id = []
        else:
            malicious_group_id = sample(group_list, args.malicious)  # 随机选择一个恶意group
        print("客户端分组情况：{}； 恶意server: {}".format(member_list, malicious_group_id))
        groups = []
        for g in group_list:
            if g in malicious_group_id:
                groups.append(MaliciousGroup(g, args, in_dim, args.device, member_list[g]))
            else:
                groups.append(CleanGroup(g, args, in_dim, args.device, member_list[g]))

        byzantine = Byzantine(args, in_dim)

        for ep in range(args.epochs):
            print("\n******************************************Epoch:{}************************************************".format(ep))
            for g in group_list:
                if g in malicious_group_id:
                    # index = malicious_group_id[malicious_group_id.index(g)-1]
                    # groups[g].server = copy.deepcopy(groups[index].server)
                    groups[g].train(logger, cleantest, attacktest)
                else:
                    groups[g].train(logger, cleantest)

            if args.aggregation == 'krum':
                selection_set = byzantine.selection(groups)
                global_client, global_server = aggregation_multi_krum(groups, malicious_group_id, selection_set)
            elif args.aggregation == 'bulyan':
                selection_set = byzantine.selection(groups)
                global_client, global_server = aggregation_bulyan(groups, malicious_group_id, selection_set)
            else:
                global_client, global_server = aggregation_average(groups, malicious_group_id)

            global_predict(ep, args, logger, in_dim, global_client, global_server, cleantest, attacktest)