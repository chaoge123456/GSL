import copy
import torch
import functools
import numpy as np
from collections import defaultdict
from network import client_resnet, server_tasker


def _krum_create_distances(users_grads):
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j])
    return distances


def krum(users_grads, users_count, corrupted_count, distances=None,return_index=False):
    if not return_index:
        assert users_count >= 2*corrupted_count + 3,('users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    non_malicious_count = users_count - corrupted_count - 2
    minimal_error = 1e30
    minimal_error_index = -1

    if distances is None:
        distances = _krum_create_distances(users_grads)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user

    if return_index:
        return minimal_error_index
    else:
        return users_grads[minimal_error_index]


def select_client(users_grads, users_count, corrupted_count, type, select):

    if type == 'bulyan':
        assert users_count >= 4 * corrupted_count + 3, 'users_count >= 4 * corrupted_count + 3'
        # assert select <= users_count - 2 * corrupted_count, 'select <= users_count - 2 * corrupted_count'
    else:
        assert users_count >= 2 * corrupted_count + 3, 'users_count >= 2 * corrupted_count + 3'
        # assert select <= users_count - 2 * corrupted_count, 'select <= users_count - 2 * corrupted_count'

    set_size = select
    # set_size = users_count - 2 * corrupted_count
    selection_set_id = []

    distances = _krum_create_distances(users_grads)
    while len(selection_set_id) < set_size:
        currently_selected = krum(users_grads, users_count - len(selection_set_id), corrupted_count, distances, True)
        selection_set_id.append(currently_selected)
        # remove the selected from next iterations:
        distances.pop(currently_selected)
        for remaining_user in distances.keys():
            distances[remaining_user].pop(currently_selected)
    print("本轮选中的更新节点为：", selection_set_id)
    return selection_set_id


class Byzantine:

    def __init__(self, args, in_dim):
        self.args = args
        self.in_dim = in_dim
        self.groups_count = args.num_group
        self.corrupted_count = args.malicious
        self.client = client_resnet(self.args.level, self.in_dim)
        self.client_weight = np.concatenate([i.data.numpy().flatten() for i in self.client.parameters()])
        self.server = server_tasker(args.level)
        self.server_weight = np.concatenate([i.data.numpy().flatten() for i in self.server.parameters()])
        self.select = {}

    def selection(self, groups):
        users_grads_client, users_grads_server = self.collect_client_weight(groups)
        selection_set_id = select_client(users_grads_client, self.groups_count, self.corrupted_count, self.args.aggregation, self.args.select)
        self.statistic(selection_set_id)
        return selection_set_id

    def collect_client_weight(self, groups):
        users_grads_client = np.empty((self.args.num_group, len(self.client_weight)), dtype=self.client_weight.dtype)
        users_grads_server = np.empty((self.args.num_group, len(self.server_weight)), dtype=self.server_weight.dtype)
        for i in range(self.args.num_group):
            users_grads_client[i, :] = np.concatenate([i.data.numpy().flatten() for i in groups[i].client.client_model.parameters()])
            users_grads_server[i, :] = np.concatenate([i.data.numpy().flatten() for i in groups[i].server.server_model.parameters()])
        return users_grads_client, users_grads_server

    def statistic(self, selection_set_id):
        for id in selection_set_id:
            if id not in self.select:
                self.select.setdefault(id, 0)
            self.select[id] += 1
        print("训练过程中每个group选中更新的次数：", self.select)

