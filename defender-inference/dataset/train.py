import copy
import torch
import torchvision
import torch.nn as nn
import numpy as np
from client import Client
import torch.nn.functional as F
from server import Server, MaliciousServer
from utils import load_client_train_data


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for v in tensor:
            for t, m, s in zip(v, self.mean, self.std):
                t.mul_(s).add_(m)
        return tensor


def get_cos_sim(v1, v2):
  num = float(np.dot(v1,v2))
  denom = np.linalg.norm(v1) * np.linalg.norm(v2)
  # return num / denom if denom != 0 else 0
  return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def zeroing_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.zeros_like(param.grad).to(param.device)


def distance_data_loss(a, b):
    l = nn.MSELoss()
    return l(a, b)


def distance_data(a, b):
    l = nn.MSELoss()
    return l(a, b)


def predict(testdata, client_model, server_model, device):
    with torch.no_grad():
        total, correct = 0, 0
        client_model.eval()
        server_model.eval()
        for data in testdata:
            x, label = data
            x, label = x.to(device), label.to(device)
            out = client_model(x)
            out = server_model(out)
            _, pred = torch.max(out, 1)
            pred = pred.view(-1)
            correct += torch.sum(torch.eq(pred, label)).item()
            total += len(label)
        accuracy = correct * 1.0 / total
        return accuracy


def reconstruct(testdata, client_model, decoder, device):
    with torch.no_grad():
        total, loss = 0, 0
        client_model.eval()
        decoder.eval()
        for data in testdata:
            x, label = data
            x, label = x.to(device), label.to(device)
            out = client_model(x)
            rec = decoder(out)
            loss += distance_data(x, rec).detach().item()
            total += 1
        avg_rec_loss = loss / total
        return avg_rec_loss


class CleanGroup:

    def __init__(self, id, args, in_dim, device, member):
        self.id = id
        self.args = args
        self.in_dim = in_dim
        self.device = device
        self.server = Server(args, in_dim)
        self.client = Client(args, in_dim)
        self.member = member
        self.count = 0

    def train(self, logger, testdata):
        self.server.server_model.to(self.device).train()
        self.client.client_model.to(self.device).train()

        for mb in self.member:
            _, client_dataset = load_client_train_data(self.args, mb)
            for (x, y) in client_dataset:
                self.server.optim.zero_grad()
                self.client.optim.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                output = self.client.client_model(x)
                output = self.server.server_model(output)
                loss = F.cross_entropy(output, y)
                loss.backward()
                self.server.optim.step()
                self.client.optim.step()

        los = loss.item()
        acc = predict(testdata, self.client.client_model, self.server.server_model, self.device)
        logger.add_scalar("Group-"+str(self.id)+"/LOSS", los, self.count)
        logger.add_scalar("Group-"+str(self.id)+"/ACCURACY", acc, self.count)
        print("Group:{}\tCount:{}\tLOSS:{:.4f}\tAccuracy:{:.4f}".format(self.id, self.count, los, acc))
        self.count += 1

        if self.args.mode == "defense":
            self.server.server_model.cpu()
            self.client.client_model.cpu()


class MaliciousGroup:

    def __init__(self, id, args, in_dim, device, member):
        self.id = id
        self.args = args
        self.in_dim = in_dim
        self.device = device
        self.server = MaliciousServer(args, in_dim)
        self.client = Client(args, in_dim)
        self.member = member
        self.count = 0

        self.client_grad = []
        self.dif_category_mean = []
        self.same_category_mean = []
        self.dif_variance = []
        self.same_variance = []

    def train(self, logger, cleantest, showimage):
        self.client.client_model.to(self.device).train()
        self.server.server_model.to(self.device).train()
        self.server.server_pilot.to(self.device).train()
        self.server.discriminator.to(self.device).train()
        self.server.server_decoder.to(self.device).train()
        attack_iterator = iter(self.server.attack_dataset)

        for mb in self.member:
            _, client_dataset = load_client_train_data(self.args, mb)
            for (x, y) in client_dataset:
                try:
                    x_public, y_public = next(attack_iterator)
                    if x_public.size(0) <= self.args.batch_size:
                        attack_iterator = iter(self.server.attack_dataset)
                        x_public, y_public = next(attack_iterator)
                except StopIteration:
                    print("sever data error")
                if x_public.size(0) > x.size(0):
                    l = x.size(0)
                    x_public, y_public = x_public[:l], y_public[:l]

                if self.args.step == "plus":
                    result = self.step_plus(x, y, x_public, y_public)
                else:
                    result = self.step(x, y, x_public, y_public)

        acc1 = predict(cleantest, self.server.server_pilot, self.server.server_model, self.device)
        acc2 = predict(cleantest, self.client.client_model, self.server.server_model, self.device)
        logger.add_scalar("Group-" + str(self.id) + "/F_LOSS", result[0], self.count)
        logger.add_scalar("Group-" + str(self.id) + "/D_LOSS", result[1], self.count)
        logger.add_scalar("Group-" + str(self.id) + "/C_LOSS", result[2], self.count)
        logger.add_scalar("Group-" + str(self.id) + "/V_LOSS", result[3], self.count)
        logger.add_scalar("Group-" + str(self.id) + "/server", acc1, self.count)
        logger.add_scalar("Group-" + str(self.id) + "/client", acc2, self.count)
        print("Group:{}\tCount:{}\tServer Accuracy:{:.4f}\tClient Accuracy:{:.4f}".format(self.id, self.count, acc1, acc2))

        self.show(self.count, logger, showimage)
        self.count += 1

        # if self.args.mode == 'attack':
        #     self.gradients_variance(y, logger, self.count)

        if self.args.mode == "defense":
            self.client.client_model.cpu()
            self.server.server_model.cpu()
            self.server.server_pilot.cpu()
            self.server.discriminator.cpu()
            self.server.server_decoder.cpu()

    def step(self, x, y, x_public, y_public):
        x = x.to(self.device)
        y = y.to(self.device)
        x_public = x_public.to(self.device)
        y_public = y_public.to(self.device)

        z_private = self.client.client_model(x)
        z_private.register_hook(self.hook)

        adv_private_logits = self.server.discriminator(z_private)
        f_loss = torch.mean(adv_private_logits)

        z_public = self.server.server_pilot(x_public)
        rec_public = self.server.server_decoder(z_public)
        c_loss = distance_data_loss(x_public, rec_public)

        adv_public_logits = self.server.discriminator(z_public.detach())
        adv_private_logits_detached = self.server.discriminator(z_private.detach())
        loss_discr_true = torch.mean(adv_public_logits)
        loss_discr_fake = -torch.mean(adv_private_logits_detached)
        D_loss = loss_discr_true + loss_discr_fake
        w = float(self.args.gp)
        D_gradient_penalty = self.gradient_penalty(z_private.detach(), z_public.detach())
        D_loss += D_gradient_penalty * w

        with torch.no_grad():
            rec_x_private = self.server.server_decoder(z_private)
            verification = distance_data(x, rec_x_private)
            loss_verification = verification.detach().item()

        self.server.optim.zero_grad()
        self.server.optim_d.zero_grad()
        self.client.optim.zero_grad()
        f_loss.backward()
        zeroing_grad(self.server.discriminator)
        c_loss.backward()
        D_loss.backward()

        self.server.optim.step()
        self.server.optim_d.step()
        self.client.optim.step()
        result = (f_loss.detach().item(), D_loss.detach().item(), c_loss.detach().item(), loss_verification)
        return result

    def step_plus(self, x, y, x_public, y_public):
        x = x.to(self.device)
        y = y.to(self.device)
        x_public = x_public.to(self.device)
        y_public = y_public.to(self.device)
        copyserver = copy.deepcopy(self.server.server_model)

        z_private = self.client.client_model(x)
        z_private.register_hook(self.hook)

        adv_private_logits = self.server.discriminator(z_private)
        client_c_private = copyserver(z_private)
        f_loss_g = torch.mean(adv_private_logits)
        f_loss_m = F.cross_entropy(client_c_private, y)
        f_loss = f_loss_g * self.args.weight_g + f_loss_m * self.args.weight_m

        z_public = self.server.server_pilot(x_public)
        c_public = self.server.server_model(z_public)
        rec_public = self.server.server_decoder(z_public)
        c_loss = F.cross_entropy(c_public, y_public) + distance_data_loss(x_public, rec_public)

        adv_public_logits = self.server.discriminator(z_public.detach())
        adv_private_logits_detached = self.server.discriminator(z_private.detach())
        loss_discr_true = torch.mean(adv_public_logits)
        loss_discr_fake = -torch.mean(adv_private_logits_detached)
        D_loss = loss_discr_true + loss_discr_fake
        w = float(self.args.gp)
        D_gradient_penalty = self.gradient_penalty(z_private.detach(), z_public.detach())
        D_loss += D_gradient_penalty * w

        with torch.no_grad():
            rec_x_private = self.server.server_decoder(z_private)
            verification = distance_data(x, rec_x_private)
            loss_verification = verification.detach().item()

        self.server.optim.zero_grad()
        self.server.optim_d.zero_grad()
        self.client.optim.zero_grad()
        f_loss.backward()
        zeroing_grad(self.server.discriminator)
        c_loss.backward()
        D_loss.backward()

        self.server.optim.step()
        self.server.optim_d.step()
        self.client.optim.step()
        result = (f_loss.detach().item(), D_loss.detach().item(), c_loss.detach().item(), loss_verification)
        return result

    def gradient_penalty(self, x, x_gen):
        epsilon = torch.rand([x.shape[0], 1, 1, 1]).cuda()
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
        from torch.autograd import grad
        d_hat = self.server.discriminator(x_hat)
        gradients = grad(outputs=d_hat, inputs=x_hat, grad_outputs=torch.ones_like(d_hat).cuda(), retain_graph=True, create_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty

    def hook(self, grad):
        self.client_grad = grad.cpu().numpy()

    def gradients_variance(self, label_private, logger, i):
        dif_category_fsha = []
        same_category_fsha = []
        gradients = self.client_grad

        num = label_private.shape[0]
        shape = gradients.reshape(num, -1).shape[1]
        for k in range(num):
            for l in range(num):
                if k != l:
                    p1 = gradients[k].reshape(shape, )
                    p2 = gradients[l].reshape(shape, )
                    if label_private[k].cpu().numpy() == label_private[l].cpu().numpy():
                        same_category_fsha.append(get_cos_sim(p1, p2))
                        # same_category_mean_.append(get_cos_sim(p1,p2))
                    else:
                        dif_category_fsha.append(get_cos_sim(p1, p2))
                        # dif_category_mean_.append(get_cos_sim(p1,p2))
        dif_category_fsha = np.array(dif_category_fsha)
        same_category_fsha = np.array(same_category_fsha)
        dif_category_mean_ = np.array(dif_category_fsha)
        same_category_mean_ = np.array(same_category_fsha)
        dif_category_mean = np.mean(dif_category_mean_)
        same_category_mean = np.mean(same_category_mean_)
        dif_variance = np.std(dif_category_mean_)
        same_variance = np.std(same_category_mean_)
        logger.add_scalar('TEST/dif_category_mean', dif_category_mean, i)
        logger.add_scalar('TEST/same_category_mean', same_category_mean, i)
        logger.add_scalar('TEST/dif_variance', dif_variance, i)
        logger.add_scalar('TEST/same_variance', same_variance, i)

        self.dif_category_mean.append(dif_category_mean)
        self.same_category_mean.append(same_category_mean)
        self.dif_variance.append(dif_variance)
        self.same_variance.append(same_variance)

    def show(self, i, logger, showimage):

        if self.in_dim == 3:
            unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        else:
            unorm = UnNormalize(mean=(0.5,), std=(0.5,))

        with torch.no_grad():
            X = copy.deepcopy(showimage)
            recovered = self.attack(X)
            X = X.detach().cpu()
            recovered = recovered.detach().cpu()
            s = "Round-" + str(i)
            # X_recovered = torch.clamp(recovered, 0, 255)
            logger.add_images(s, unorm(recovered))

    def attack(self, x_private):
        with torch.no_grad():
            # smashed data sent from the client:
            z_private = self.client.client_model(x_private)
            # recover private data from smashed data
            tilde_x_private = self.server.server_decoder(z_private)
            # z_private_control = self.server.server_pilot(x_private)
            # control = self.server.server_decoder(z_private_control)
            return tilde_x_private









