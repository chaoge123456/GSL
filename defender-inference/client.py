import torch
import torch.nn as nn
from network import client_resnet


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()


class Client:

    def __init__(self, args, in_dim):
        self.args = args
        self.in_dim = in_dim
        self.client_model = client_resnet(args.level, in_dim)
        self.client_model.apply(init_weights)
        self.optim = torch.optim.Adam(self.client_model.parameters(), lr=args.lr_client)