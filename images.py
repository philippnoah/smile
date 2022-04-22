import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributions as dist
import os
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * config.repeat * config.channels, 64, 5, 2, 2),
            nn.ReLU()
        )
        dim_x, dim_y = utils.next_conv_size(dim_x, dim_y, 5, 2, 2)

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

def generate_mask(config):
    batch_size = config.batch_size
    channels = config.channels
    x_dim, y_dim = config.dims[:2]
    mask = config.mask
    return torch.cat([
        torch.ones(batch_size, channels, x_dim - mask, x_dim),
        torch.zeros(batch_size, channels, mask, x_dim)
    ], dim=2).to(device)

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
        o = torch.cat(processed_img, dim=0).to(device)
        return o

    return f

def generate_test(X, imgs, masks, dim, transform):
    batch_size, channels = X[0].size(0), X[0].size(1)
    x = [X[i].to(device) for i in imgs[0]]
    y = [X[i].to(device) for i in imgs[1]]

    if transform == 'mask':
        mx = [generate_mask(batch_size, channels, m, dim) for m in masks[0]]
        my = [generate_mask(batch_size, channels, m, dim) for m in masks[1]]

        x = torch.cat(x, dim=1) * torch.cat(mx, dim=1)
        y = torch.cat(y, dim=1) * torch.cat(my, dim=1)
    else:
        x = torch.cat(x, dim=1)
        mask = masks[0]
        t = generate_transform(batch_size, channels, mask, dim, transform)
        my = [t(b) for b in y]
        y = torch.cat(my, dim=1)

    return x, y

def MI(f, config, **kwargs):
    if config.estimator_name == 'infonce':
        loss = -utils.infonce_lower_bound(f)
    elif config.estimator_name == 'dv':
        loss = -utils.dv_upper_lower_bound(f)
    elif config.estimator_name == 'nwj':
        loss = -utils.nwj_lower_bound(f)
    elif config.estimator_name == 'reg_dv':
        loss = -utils.regularized_dv_bound(f, **kwargs)
    elif config.estimator_name == 'smile':
        loss = -utils.smile_lower_bound(f, **kwargs)
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
    else:
        normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        dataset = datasets.CIFAR10('data/cifar/', train=True, download=True, \
            transform=transforms.Compose([transf, normal]))

    random_indices = np.random.permutation(np.arange(0, int(len(dataset) * config.dataset_ratio)))

    dataset = td.Subset(dataset, random_indices)

    train_dataloader = td.DataLoader(dataset, batch_size=config.batch_size * config.repeat, \
        shuffle=True, num_workers=2, pin_memory=True)
        
    test_dataloader = td.DataLoader(dataset, batch_size=config.batch_size * config.repeat, \
        shuffle=True, num_workers=2, pin_memory=True)

    return train_dataloader, test_dataloader

def test(test_dataloader, net, config, **kwargs):
    with torch.no_grad():
        test_losses = []
        for j, x in enumerate(test_dataloader):
            x = torch.chunk(x[0], config.repeat, dim=0)
            x, y = generate_test(x, config)
            f = net(x, y).to(device)
            loss = MI(f, buffer=config.buffer, **kwargs)

            if j % 20 == 0 and config.debug:
                print(f'{j}: {-loss.item():.3f}')
                if config.estimator_name == 'vae':
                    print(f[0].mean().item(), f[1].mean().item())

            mi = -loss.item()
            test_losses.append(mi)
    return test_losses

def init_estimator(config):
    if config.estimator_name == 'vae':
        pnet = VAE(config, repeat=config.repeat * 2)
        qnet1 = VAE(config)
        qnet2 = VAE(config)
        net = VAEConcatCritic([pnet, qnet1, qnet2])
        net.to(device)
    else:
        conv_net = ConvNet(config)
        net = ConcatCritic(conv_net)
        net.to(device)
    return net

def train(train_dataloader, net, config, **kwargs):
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    losses = []
    for i in range(config.epochs):
        for j, x in enumerate(train_dataloader):
            optimizer.zero_grad()
            bs = x[0].size(0)
            if bs != config.batch_size * config.repeat:
                pass
            x = torch.chunk(x[0], config.repeat, dim=0)
            x, y = generate_test(x, config)
            f = net(x, y).to(device)
            loss = MI(f, **kwargs)
            loss.backward()

            if j % 20 == 0 and config.debug:
                print(f'{j}: {-loss.item():.3f}')
                if config.estimator_name == 'vae':
                    print(f[0].mean().item(), f[1].mean().item())

            optimizer.step()
            mi = -loss.item()
            losses.append(mi)

def image_mi_estimator(config, **kwargs):
    train_dataloader, test_dataloader = load_dataset(config)
    net = init_estimator(config)
    train_losses = train(train_dataloader, net, config, **kwargs)
    test_losses = test(test_dataloader, net, config, **kwargs)
    return np.array(losses), np.array(test_losses)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator-name', type=str, default='smile')
    parser.add_argument('--imgs', type=str, default=None)
    parser.add_argument('--masks', type=str, default=None)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dataset-name', type=str, default='mnist')
    parser.add_argument('--transform', type=str, default='mask')
    parser.add_argument('--dataset-ratio', type=float, default=1.0)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    return args

def init(config):
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

def save_losses(config):
    logdir_t = f'logs/{config.dataset}/{config.exp}_t'
    logdir = f'logs/{config.dataset}/{config.exp}'
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(logdir_t, exist_ok=True)
    savedir = os.path.join(logdir, f'{config.critic}_{config.imgs}_{config.masks}')
    np.save(savedir, losses)
    savedir = os.path.join(logdir_t, f'{config.critic}_{config.imgs}_{config.masks}')
    np.save(savedir, test_losses)

if __name__ == '__main__':
    config = parse_args()
    init(config)
    losses, test_losses = image_mi_estimator(config)
    save_losses()