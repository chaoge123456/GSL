import copy
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from network import server_pilot, server_tasker, discriminator, server_dis, Discriminator_server, server_decoder


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()


def addtrigger(img):

    _, h, w = img.shape
    img[:, h-27:h-24, w-27:w-24] = 1
    return img


class Backdoor_train(Dataset):

    def __init__(self, data, target, index_backdoor, in_dim=1):

         self.data = data
         self.target = target
         self.index_backdoor = index_backdoor
         self.in_dim = in_dim

    def __getitem__(self, index):

        if index in self.index_backdoor:
            clean = self.data[index][0]
            image = addtrigger(copy.deepcopy(clean)).numpy()
            label = torch.tensor(self.target)
            is_backdoor = torch.tensor(1)
            true_label = torch.tensor(self.data[index][1])
        else:
            image = self.data[index][0].numpy()
            label = torch.tensor(self.data[index][1])
            is_backdoor = torch.tensor(0)
            true_label = torch.tensor(self.data[index][1])
        return image, label, is_backdoor, true_label

    def __len__(self):

        return len(self.data)


class Server:

    def __init__(self, args, in_dim):

        self.args = args
        self.in_dim = in_dim
        self.server_model = server_tasker(args.level)
        self.server_model.apply(init_weights)
        self.optim = torch.optim.Adam(self.server_model.parameters(), lr=args.lr_server)


class MaliciousServer:

    def __init__(self, args, in_dim):

        self.args = args
        self.in_dim = in_dim
        self.server_pilot = server_pilot(args.level, in_dim)
        self.server_model = server_tasker(args.level)
        self.server_dis = server_dis(args.level)
        if args.step == "embed":
            self.discriminator = Discriminator_server(args.level)
        else:
            self.discriminator = discriminator(args.level)

        self.server_pilot.apply(init_weights)
        self.server_model.apply(init_weights)
        self.discriminator.apply(init_weights)
        self.server_dis.apply(init_weights)
        self.optim = torch.optim.Adam([{'params': self.server_pilot.parameters()}, {'params': self.server_model.parameters()}, {'params': self.server_dis.parameters()}], lr=args.lr_server)
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr_D)
        self.attack_dataset = self.serverdata()

    def serverdata(self):

        dir_path = "dataset/" + self.args.dataset + "/"
        if self.args.dataset == "mnist":
            transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(32, padding=4), transforms.Normalize([0.5], [0.5])])
            data = torchvision.datasets.MNIST(root=dir_path + "rawdata", train=True, download=False, transform=transform)

        if self.args.dataset == "fmnist":
            transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(32, padding=4), transforms.Normalize([0.5], [0.5])])
            data = torchvision.datasets.FashionMNIST(root=dir_path + "rawdata", train=True, download=False, transform=transform)

        if self.args.dataset == "cifar10":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = torchvision.datasets.CIFAR10(root=dir_path + "rawdata", train=True, download=False, transform=transform)

        if self.args.dataset == "cifar100":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            data = torchvision.datasets.CIFAR100(root=dir_path + "rawdata", train=True, download=False, transform=transform)

        shuffle = torch.randperm(len(data)).tolist()[:self.args.server_sample]
        data_ = copy.deepcopy(data)
        server_data = [data_[i] for i in shuffle]
        index_backdoor = torch.randperm(len(server_data))[:self.args.num_backdoor]  # 随机选择部分样本添加后门
        backdoor_data = Backdoor_train(server_data, self.args.target, index_backdoor, self.in_dim)
        return DataLoader(backdoor_data, self.args.batch_size, shuffle=True)


