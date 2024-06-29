import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import GreedyHash_get_config
from model.data import my_dataloader
from model.vit import CONFIGS, VisionTransformer
from utils.utils import evalModel, save_config
from model.dcgan import DCGAN_MODEL,Discriminator,Generator
from torch.autograd import Variable
import torch.optim as optim
from configs.utils import config_dataset
from model.data import CIB_CIFAR_DataLoader, get_data_CIB
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'

start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

# 模型配置
config_ = {
    # "dataset": "mirflickr",
    "dataset": "cifar10-1",
    # "dataset": "coco",
    # "dataset": "nuswide_21",
    # "dataset": "nuswide_10",

    "bit_list": [12, 24, 32, 48, 64],

    "info": "VBGAN",
    "backbone": "ViT-B_16",
    "pretrained_dir": "checkpoint/ViT-B_16.npz",

    "frozen backbone": True,
    # "optimizer": {"type": optim.Adam,
    #               "lr": 0.001,
    #               "backbone_lr": 1e-5},
    "optimizer": {"type": optim.Adam,
                  "lr": 0.001,
                  "backbone_lr": 1e-5},
    "epoch": 20,
    "test_map": 2,
    "batch_size": 64,
    "num_workers": 8,
    "logs_path": "logs",

    "resize_size": 224,
    "crop_size": 224,

    "temperature": 0.3,
    "weight": 0.001,
    "channels": 3,
    "cuda": True
}
config = config_dataset(config_)
config["logs_path"] = os.path.join(config["logs_path"], config['info'], start_time)

if not os.path.exists(config["logs_path"]):
    os.makedirs(config["logs_path"])

if 'cifar' in config["dataset"]:
    config["topK"] = 1000
else:
    config["topK"] = 5000


class NtXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NtXentLoss, self).__init__()
        self.temperature = temperature

        self.similarityF = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j, device):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # sim = 0.5 * (z_i.shape[1] - torch.tensordot(z.unsqueeze(1), z.T.unsqueeze(0), dims = 2)) / z_i.shape[1] / self.temperature

        sim_i_j = torch.diag(sim, batch_size)

        sim_j_i = torch.diag(sim, -batch_size)

        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N).cuda().long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class TDCBGANModel(nn.Module):
    def __init__(self, bit, config):
        super(TDCBGANModel, self).__init__()

        # ViT模型配置
        vit_config = CONFIGS[config['backbone']]
        vit_config.pretrained_dir = config['pretrained_dir']
        self.batch_size = config['batch_size']
        self.vit = VisionTransformer(vit_config, 224, num_classes=1000, zero_head=False, vis=True)
        self.vit.load_from(np.load(vit_config.pretrained_dir))

        for param in self.vit.parameters():
            param.requires_grad = True

        # 全连接层
        self.fc_encode = nn.Linear(vit_config.hidden_size, bit)

        # DCGAN网络
        self.G = Generator(config['channels'],bit)
        self.D = Discriminator(config['channels'])
        self.C = config['channels']

        # 优化器和损失函数
        self.loss = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.cuda = False
        self.cuda_index = 0
        # check if cuda is available
        self.check_cuda(config['cuda'])

        self.TDCBGAN_optimizer = config["optimizer"]["type"]([{"params": self.fc_encode.parameters(), "lr": config["optimizer"]["lr"]},
                                             {"params": self.vit.parameters(),
                                              "lr": config["optimizer"]["backbone_lr"]}])

        # 超参数
        self.beta = 5
        self.epoch = 1
        self.theta = 0.1
        self.sita = 0.5

        self.kl_weight = config['weight']
        self.criterion = NtXentLoss(config['batch_size'], config['temperature'])

    # Hash层
    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(_, input):
            return input.sign()

        @staticmethod
        def backward(_, grad_output):
            return grad_output

    def forward(self, img_i, img_j=None, img_gan=None, device=None, is_train=True):
        # 生成Hash码
        if not is_train:
            x, _ = self.vit(img_i)
            prob = torch.sigmoid(self.fc_encode(x[:,0]))
            z = TDCBGANModel.Hash.apply(prob - 0.5)
            return z
        # 训练
        self.TDCBGAN_optimizer.zero_grad()
        x, _ = self.vit(img_i)
        h = self.fc_encode(x[:,0])

        # CIB 损失
        prob_i = torch.sigmoid(h)
        z_i = TDCBGANModel.Hash.apply(prob_i - 0.5)

        imgj, _ = self.vit(img_j)
        prob_j = torch.sigmoid(self.fc_encode(imgj[:,0]))
        z_j = TDCBGANModel.Hash.apply(prob_j - 0.5)

        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2
        contra_loss = self.criterion(z_i, z_j, device)
        CIB_loss = contra_loss + self.kl_weight * kl_loss

        CIB_loss.backward()
        self.TDCBGAN_optimizer.step()
        torch.cuda.empty_cache()

        # DCGAN 损失
        real_labels = torch.ones(img_gan.shape[0])
        fake_labels = torch.zeros(img_gan.shape[0])
        x, _ = self.vit(img_i)
        z = self.fc_encode(x[:,0])
        features = (torch.sigmoid(self.beta * z) - 0.5) * 2
        features = torch.reshape(features, (features.shape[0], features.shape[1], 1, 1))


        if self.cuda:
            images, features = Variable(img_gan).cuda(self.cuda_index), Variable(features).cuda(self.cuda_index)
            real_labels, fake_labels = Variable(real_labels).cuda(self.cuda_index), Variable(fake_labels).cuda(
                self.cuda_index)
        else:
            images, features = Variable(img_gan), Variable(features)
            real_labels, fake_labels = Variable(real_labels), Variable(fake_labels)

        # 判别器损失
        outputs = self.D(images)
        d_loss_real = self.loss(outputs.flatten(), real_labels)

        fake_images = self.G(features)
        outputs = self.D(fake_images)
        d_loss_fake = self.loss(outputs.flatten(), fake_labels)

        d_loss = self.sita * (d_loss_real + d_loss_fake)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        torch.cuda.empty_cache()

        # 生成器损失
        fake_images = self.G(features)
        outputs = self.D(fake_images)
        g_loss = self.sita * self.loss(outputs.flatten(), real_labels)

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        self.TDCBGAN_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        torch.cuda.empty_cache()

        # 内容损失
        x, _ = self.vit(img_i)
        h = self.fc_encode(x[:, 0])
        features = (torch.sigmoid(self.beta * h) - 0.5) * 2
        features = torch.reshape(features, (features.shape[0], features.shape[1], 1, 1))
        fake_images = self.G(features)
        p_loss = nn.MSELoss()

        # precision_loss = p_loss(x_fake, x)
        content_loss = p_loss(fake_images, images)
        con_loss = self.theta * content_loss
        con_loss.backward()
        self.TDCBGAN_optimizer.step()
        self.g_optimizer.step()
        torch.cuda.empty_cache()

        return d_loss, g_loss, CIB_loss, con_loss

    def encode_discrete(self, x):
        x, _ = self.vit(x)
        prob = torch.sigmoid(self.fc_encode(x))
        z = TDCBGANModel.Hash.apply(prob - 0.5)

        return z

    def modifier(self):
        self.beta += 0
        self.epoch += 1
        # print(self.beta)

    def modifier_beta(self):
        self.beta = 5
        self.epoch = 1

    def compute_kl(self, prob, prob_v):
        prob_v = prob_v.detach()
        # prob = prob.detach()
        kl = prob * (torch.log(prob + 1e-8) - torch.log(prob_v + 1e-8)) + (1 - prob) * (torch.log(1 - prob + 1e-8 ) - torch.log(1 - prob_v + 1e-8))
        kl = torch.mean(torch.sum(kl, axis = 1))
        return kl

    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            self.loss = nn.BCELoss().cuda(self.cuda_index)
            print("Cuda enabled flag: ")
            print(self.cuda)


def trainer(config, bit):
    Best_mAP = 0

    train_logfile = open(os.path.join(config['logs_path'], 'train_log.txt'), 'a')
    train_logfile.write(f"***** {config['info']} - {config['backbone']} - {bit}bit *****\n\n")

    """DataLoader"""
    if "cifar" in config['dataset']:
        data = CIB_CIFAR_DataLoader(config['dataset'])
        train_loader, test_loader, _, database_loader, num_train, num_test, num_database = data.get_loaders(
            config['batch_size'], 8,
            shuffle_train=True, get_test=False
        )
    else:
        train_loader, test_loader, database_loader, num_train, num_test, num_database = get_data_CIB(config)

    """Model"""
    device = torch.device('cuda')
    net = TDCBGANModel(bit, config)
    net = net.to(device)

    """Data Parallel"""
    net = torch.nn.DataParallel(net)

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s-%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], config["backbone"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]),
              end="")
        net.train()
        train_loss = 0
        cib_loss_ = 0
        con_loss_ = 0
        d_loss_ = 0
        g_loss_ = 0
        # 训练
        for image1, image2, image_gan in train_loader:
            image1 = image1.to(device)
            image2 = image2.to(device)
            image_gan = image_gan.to(device)

            d_loss, g_loss, cib_loss, con_loss = net(image1, image2, image_gan, device)
            d_loss_ += d_loss
            g_loss_ += g_loss
            cib_loss_ += cib_loss
            con_loss_ += con_loss
            train_loss += d_loss + g_loss + cib_loss + con_loss

        train_loss = train_loss / len(train_loader)
        d_loss_ = d_loss_ / len(train_loader)
        g_loss_ = g_loss_ / len(train_loader)
        cib_loss_ = cib_loss_ / len(train_loader)
        con_loss_ = con_loss_ / len(train_loader)


        print("\b\b\b\b\b\b\b loss:%.5f | cib_loss:%.5f | con_loss:%.5f" % (train_loss, cib_loss_, g_loss_))
        train_logfile.write(
            'Train | %s-%s[%2d/%2d][%s] bit:%d, dataset:%s | Loss: %.5f | CIB Loss: %.5f | Con Loss: %.5f | g Loss: %.5f | d Loss: %.5f \n' %
            (config["info"], config["backbone"], epoch + 1, config["epoch"], current_time, bit, config["dataset"],
             train_loss, cib_loss_, con_loss_, g_loss_, d_loss_))

        # 调试
        if (epoch + 1) % config["test_map"] == 0:
            net.eval()
            if epoch + 1 != 0:
                with torch.no_grad():
                    Best_mAP = evalModel(test_loader, database_loader, net, Best_mAP, bit, config, epoch + 1,
                                         train_logfile)

        if (epoch + 1) % 1 == 0:
            net.module.modifier()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    start_time = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
    setup_seed(2022)

    save_config(config, config["logs_path"])
    # print(config)
    for bit in config["bit_list"]:
        trainer(config, bit)
