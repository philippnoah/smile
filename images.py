import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributions as dist
import os
import matplotlib.pyplot as plt
import utils
from datetime import datetime as dt

class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        x_dim, y_dim = config.dims[:2]
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * config.repeat * config.channels, 64, 5, 2, 2),
            nn.ReLU()
        )
        dim_x, dim_y = utils.next_conv_size(x_dim, y_dim, 5, 2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU()
        )
        dim_x, dim_y = utils.next_conv_size(dim_x, dim_y, 5, 2, 2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * dim_x * dim_y, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.fc(z)
        return z

class VAE(nn.Module):
    def __init__(self, config, repeat=None):
        super(VAE, self).__init__()
        self.config = config
        x_dim, y_dim, z_dim = config.dims
        channels = config.channels
        repeat = repeat or config.repeat

        self.conv1 = nn.Sequential(
            nn.Conv2d(repeat * channels, 64 * repeat, 5, 2, 2),
            nn.ReLU()
        )
        dim_x, dim_y = utils.next_conv_size(x_dim, y_dim, 5, 2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * repeat, 128 * repeat, 5, 2, 2),
            nn.ReLU()
        )
        dim_x, dim_y = utils.next_conv_size(dim_x, dim_y, 5, 2, 2)
        self.top_conv_x, self.top_conv_y = dim_x, dim_y
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * repeat * dim_x * dim_y, 1024 * repeat),
            nn.ReLU(),
            nn.Linear(1024 * repeat, z_dim * 2 * repeat)
        )
        self.enc = nn.Sequential(
            self.conv1, self.conv2, self.fc
        )

        if self.dim_x == 28:
            self.dec = nn.Sequential(
                nn.Linear(z_dim * repeat, 400 * repeat),
                nn.Softplus(),
                nn.Linear(400 * repeat, 400 * repeat),
                nn.Softplus(),
                nn.Linear(400 * repeat, 784 * repeat)
            )
        else:
            self.dec_fc = nn.Sequential(
                nn.Linear(z_dim * repeat, 1024 * repeat),
                nn.ReLU(),
                nn.Linear(1024 * repeat, 128 * repeat * dim_x * dim_y),
                nn.ReLU()
            )
            self.dec_deconv = nn.Sequential(
                nn.ConvTranspose2d(128 * repeat, 64 * repeat, 5, 2, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64 * repeat, repeat * channels, 5, 2, 2, 1)
            )

    def forward(self, x):
        z = self.enc(x)
        z_dims = self.z_dim * self.repeat
        z_m, log_z_s = z[:, :z_dims], z[:, z_dims:]

        kl = torch.sum(-log_z_s + (log_z_s.exp() **
                                   2 + z_m ** 2) / 2.0 - 0.5, dim=1)
        epsilon = torch.randn_like(z_m)
        if self.dim_x == 28:
            x_ = self.dec(z_m + epsilon * torch.exp(log_z_s)
                          ).view(-1, self.repeat, 28, 28)
            rec = torch.nn.BCEWithLogitsLoss(
                reduction='none')(x_, x).sum(dim=(1, 2, 3))
        else:
            x_ = self.dec_fc(z_m + epsilon * torch.exp(log_z_s)
                             ).view(-1, self.repeat * 128, self.top_conv_x, self.top_conv_y)
            x_ = x_.view(-1, 128 * self.repeat,
                         self.top_conv_x, self.top_conv_y)
            x_ = self.dec_deconv(x_)
            rec = torch.nn.BCEWithLogitsLoss(
                reduction='none')(x_, x).sum(dim=(1, 2, 3))

        return - rec - kl

class ConcatCritic(nn.Module):
    def __init__(self, net):
        super(ConcatCritic, self).__init__()
        self.net = net

    def forward(self, x, y):
        batch_size = x.size(0)
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, x.size(1) * 2] + list(x.shape[2:]))
        scores = self.net(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()

class VAEConcatCritic(nn.Module):
    def __init__(self, net):
        super(VAEConcatCritic, self).__init__()
        self.p, self.q1, self.q2 = net

    def forward(self, x, y):
        xy_concat = torch.cat((x, y), dim=1)
        fp = self.p(xy_concat)
        fq1 = self.q1(x)
        fq2 = self.q2(y)
        fq = fq1.view(-1, 1) + fq2.view(1, -1)
        return fp, fq

def generate_mask(mask, config):
    batch_size = config.batch_size
    channels = config.channels
    x_dim, y_dim = config.dims[:2]
    return torch.cat([
        torch.ones(batch_size, channels, x_dim - mask, x_dim),
        torch.zeros(batch_size, channels, mask, x_dim)
    ], dim=2).to(config.device)

def generate_transform(config):
    x_dim = config.dims[0]
    channels = config.channels

    if config.transform == 'rotation':
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
        ])
    elif config.transform == 'translation':
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(0, translate=[0.1, 0.1]),
            transforms.Resize(x_dim),
            transforms.ToTensor(),
        ])
    elif config.transform == 'scaling':
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(0, scale=[1.2, 1.2]),
            transforms.Resize(x_dim),
            transforms.ToTensor(),
        ])

    def f(batch_img):
        processed_img = []
        for i in range(batch_img.size(0)):
            processed_img.append(
                t(batch_img[i].cpu()).view(1, channels, x_dim, x_dim))
        o = torch.cat(processed_img, dim=0).to(config.device)
        return o

    return f

def generate_test(X, config):
    x = [X[i].to(config.device) for i in config.imgs[0]]
    y = [X[i].to(config.device) for i in config.imgs[1]]

    if config.transform == 'mask':
        mx = [generate_mask(m, config) for m in config.masks[0]]
        my = [generate_mask(m, config) for m in config.masks[1]]

        x = torch.cat(x, dim=1) * torch.cat(mx, dim=1)
        y = torch.cat(y, dim=1) * torch.cat(my, dim=1)
    else:
        x = torch.cat(x, dim=1)
        t = generate_transform(config)
        my = [t(b) for b in y]
        y = torch.cat(my, dim=1)

    return x, y

def MI(f, config):
    if config.estimator_name == 'infonce':
        loss = -utils.infonce_lower_bound(f)
    elif config.estimator_name == 'dv':
        loss = -utils.dv_upper_lower_bound(f)
    elif config.estimator_name == 'nwj':
        loss = -utils.nwj_lower_bound(f)
    elif config.estimator_name == 'reg_dv':
        loss = -utils.regularized_dv_bound(f, l=config.l)
    elif config.estimator_name == 'smile':
        loss = -utils.smile_lower_bound(f, alpha=config.alpha, clip=config.clip)
    elif config.estimator_name == 'mine':
        loss, config.buffer = utils.mine_lower_bound(f, buffer=config.buffer, momentum=0.9)
        loss = -loss
    elif config.estimator_name == 'vae':
        loss = utils.vae_lower_bound(f)
        loss = -loss
    return loss

def load_dataset(config):
    transf = transforms.ToTensor()
    if config.dataset_name == 'mnist':
        dataset = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([transf]))
        config.channels = 1
        config.dims = (28, 28)
    else:
        normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        dataset = datasets.CIFAR10('data/cifar/', train=True, download=True, \
            transform=transforms.Compose([transf, normal]))
        config.channels = 3
        config.dims = (32, 32)

    random_indices = np.random.permutation(np.arange(0, int(len(dataset) * config.dataset_ratio)))

    dataset = td.Subset(dataset, random_indices)

    train_dataloader = td.DataLoader(dataset, batch_size=config.batch_size * config.repeat, \
        shuffle=True, num_workers=2, pin_memory=True)
        
    test_dataloader = td.DataLoader(dataset, batch_size=config.batch_size * config.repeat, \
        shuffle=True, num_workers=2, pin_memory=True)

    return train_dataloader, test_dataloader

def test(test_dataloader, net, config, **kwargs):
    test_losses = []
    with torch.no_grad():
        for j, x in enumerate(test_dataloader):
            if j > config.iterations:
                break
            x = torch.chunk(x[0], config.repeat, dim=0)
            x, y = generate_test(x, config)
            f = net(x, y).to(config.device)
            loss = MI(f, config)

            if j % 20 == 0 and config.debug:
                print(f'{j}: {-loss.item():.3f}')
                if config.estimator_name == 'vae':
                    print(f[0].mean().item(), f[1].mean().item())

            mi = -loss.item()
            test_losses.append(mi)
    return np.array(test_losses)

def init_estimator(config):
    if config.estimator_name == 'vae':
        pnet = VAE(config, repeat=config.repeat * 2)
        qnet1 = VAE(config)
        qnet2 = VAE(config)
        net = VAEConcatCritic([pnet, qnet1, qnet2])
        net.to(config.device)
    else:
        config.channels = 1
        conv_net = ConvNet(config)
        net = ConcatCritic(conv_net)
        net.to(config.device)
    return net

def train(train_dataloader, net, config, **kwargs):
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    train_losses = []
    for i in range(config.epochs):
        for j, x in enumerate(train_dataloader):
            if j > config.iterations:
                break
            optimizer.zero_grad()
            x = torch.chunk(x[0], config.repeat, dim=0)
            x, y = generate_test(x, config)
            f = net(x, y).to(config.device)
            loss = MI(f, config)
            loss.backward()

            if j % 20 == 0 and config.debug:
                print(f'{j}: {-loss.item():.3f}')
                if config.estimator_name == 'vae':
                    print(f[0].mean().item(), f[1].mean().item())

            optimizer.step()
            mi = -loss.item()
            train_losses.append(mi)
    return np.array(train_losses)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator-name', type=str, default='smile')
    parser.add_argument('--imgs', type=str, default=None)
    parser.add_argument('--masks', type=str, default=None)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dataset-name', type=str, default='mnist')
    parser.add_argument('--transform', type=str, default='mask')
    parser.add_argument('--dataset-ratio', type=float, default=1.0)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--iterations', type=float, default=20)#float("inf"))
    parser.add_argument('--fig-dir', type=str, default='figs/')
    args = parser.parse_args()
    return args

def init_config():
    config = parse_args()

    if config.imgs:
        config.imgs = eval(config.imgs)

    if config.masks:
        config.masks = eval(config.masks)

    if config.imgs is None:
        config.imgs = ([0], [0])

    if config.masks is None:
        config.masks = ([0], [0])

    config.repeat = len(config.imgs[0])

    if config.estimator_name == 'vae':
        config.num_epochs = 10 * config.repeat

    config.clip = None
    if '_' in config.estimator_name:
        config.clip = float(config.estimator.split('_')[-1])

    config.buffer = None
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config.log_dir_t = f'logs/{config.dataset_name}/{config.exp}_t'
    config.log_dir = f'logs/{config.dataset_name}/{config.exp}'

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.log_dir_t, exist_ok=True)
    
    config.filename = (
        f"{config.estimator_name}_"
        f"{str(config.imgs).replace(' ', '')}_"
        f"{str(config.masks).replace(' ', '')}"
        )

    if config.debug:
        config.filename = f"{config.estimator_name}_{dt.now().strftime('%Y-%m-%d-_%H:%M:%S')}"

    config.loss_log_path = os.path.join(config.log_dir, config.filename)
    config.loss_log_path_t = os.path.join(config.log_dir_t, config.filename)

    os.makedirs(config.fig_dir, exist_ok=True)
    config.fig_file_path = os.path.join(config.fig_dir, config.filename)

    if config.debug:
        print(config)

    return config

def plot(losses, config, label):
    plt.plot(losses, label=label)
    plt.title(config)
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('MI')
    path = f"{config.fig_file_path}_{label}"
    plt.savefig(path)
    plt.clf()

def save_losses(losses, test_losses, config):
    np.save(config.loss_log_path, losses)
    np.save(config.loss_log_path_t, test_losses)

    plot(train_losses, config, label='train')
    plot(test_losses, config, label='test')

if __name__ == '__main__':
    # initialize
    config = init_config()
    train_dataloader, test_dataloader = load_dataset(config)
    net = init_estimator(config)

    # train, test
    train_losses = train(train_dataloader, net, config)
    test_losses = test(test_dataloader, net, config)

    # save results
    save_losses(train_losses, test_losses, config)