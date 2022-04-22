import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributions as dist

import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def next_conv_size(dim_x, dim_y, k, s, p):
    '''Infers the next size of a convolutional layer.
    Args:
        dim_x: First dimension.
        dim_y: Second dimension.
        k: Kernel size.
        s: Stride.
        p: Padding.
    Returns:
        (int, int): (First output dimension, Second output dimension)
    '''

    def infer_conv_size(w, k, s, p):
        '''Infers the next size after convolution.
        Args:
            w: Input size.
            k: Kernel size.
            s: Stride.
            p: Padding.
        Returns:
            int: Output size.
        '''
        x = (w - k + 2 * p) // s + 1
        return x

    if isinstance(k, int):
        kx, ky = (k, k)
    else:
        kx, ky = k

    if isinstance(s, int):
        sx, sy = (s, s)
    else:
        sx, sy = s

    if isinstance(p, int):
        px, py = (p, p)
    else:
        px, py = p
    return (infer_conv_size(dim_x, kx, sx, px),
            infer_conv_size(dim_y, ky, sy, py))


def next_deconv_size(dim_x, dim_y, k, s, p):
    def infer_conv_size(w, k, s, p):
        x = (w - 1) * s - 2 * p + k


class ConvNet(nn.Module):
    def __init__(self, repeat=1, channels=1, dim_x=28, dim_y=28):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * repeat * channels, 64, 5, 2, 2),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        dim_x, dim_y = next_conv_size(dim_x, dim_y, 5, 2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, 2),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )
        dim_x, dim_y = next_conv_size(dim_x, dim_y, 5, 2, 2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * dim_x * dim_y, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight, gain=repeat)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.fc(z)
        return z


class VAE(nn.Module):
    def __init__(self, repeat=1, z_dim=10, channels=1, dim_x=28, dim_y=28):
        super(VAE, self).__init__()
        self.repeat = repeat
        self.channels = channels
        self.z_dim = z_dim
        self.dim_x, self.dim_y = dim_x, dim_y

        self.conv1 = nn.Sequential(
            nn.Conv2d(repeat * channels, 64 * repeat, 5, 2, 2),
            nn.ReLU()
        )
        dim_x, dim_y = next_conv_size(dim_x, dim_y, 5, 2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * repeat, 128 * repeat, 5, 2, 2),
            nn.ReLU()
        )
        dim_x, dim_y = next_conv_size(dim_x, dim_y, 5, 2, 2)
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
        # print('rk:', rec.mean().item(), kl.mean().item())

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
        batch_size = x.size(0)
        # x_tiled = torch.stack([x] * batch_size, dim=0)
        # y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
        #                          batch_size * batch_size, x.size(1) * 2] + list(x.shape[2:]))
        xy_concat = torch.cat((x, y), dim=1)
        fp = self.p(xy_concat)
        fq1 = self.q1(x)
        fq2 = self.q2(y)
        fq = fq1.view(-1, 1) + fq2.view(1, -1)
        # fq1 = self.q(xy_pairs).view([batch_size, batch_size]).t()

        return fp, fq


def generate_mask(batch_size, channels, mask, dim=28):
    return torch.cat([
        torch.ones(batch_size, channels, dim - mask, dim),
        torch.zeros(batch_size, channels, mask, dim)
    ], dim=2).to(device)


def generate_transform(batch_size, channels, mask, dim, transform):
    if transform == 'rotation':
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ])
    elif transform == 'translation':
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(0, translate=[0.1, 0.1]),
            # transforms.RandomCrop(
            # dim, padding=4, fill=0),
            transforms.Resize(dim),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ])
    elif transform == 'scaling':
        t = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(0, scale=[1.2, 1.2]),
            transforms.Resize(dim),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ])

    def f(batch_img):
        processed_img = []
        for i in range(batch_img.size(0)):
            processed_img.append(
                t(batch_img[i].cpu()).view(1, channels, dim, dim))
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
        # import ipdb
        # ipdb.set_trace()
        y = torch.cat(my, dim=1)

    x, y = x.to(device), y.to(device)
    return x, y


def image_mi_estimator(dataset_name='mnist', dataset_ratio=1.0, critic_type='smile', imgs=None,
                       masks=None, debug=False, test=False, transform='mask', **kwargs):
    if dataset_name == 'mnist':
        dataset = datasets.MNIST('data/', train=True, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))
        channels = 1
        dim_x = 28
        dim_y = 28
    else:
        dataset = datasets.CIFAR10('data/cifar/', train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
        channels = 3
        dim_x = 32
        dim_y = 32

    random_indices = np.random.permutation(np.arange(0, len(dataset)))[
        :int(len(dataset) * dataset_ratio)]

    if imgs is None:
        imgs = ([0], [0])
    if masks is None:
        masks = ([0], [0])

    repeat = len(imgs[0])

    dataset = td.Subset(dataset, random_indices)
    batch_size = 64
    loaders = td.DataLoader(dataset, batch_size=batch_size * repeat, shuffle=True,
                            num_workers=2, pin_memory=True)
    if critic_type != 'vae':
        net = ConcatCritic(
            ConvNet(repeat, channels=channels, dim_x=dim_x, dim_y=dim_y))
        net.to(device)
    else:
        pnet = VAE(repeat=repeat * 2, channels=channels,
                   dim_x=dim_x, dim_y=dim_y)
        qnet1 = VAE(repeat=repeat, channels=channels, dim_x=dim_x, dim_y=dim_y)
        qnet2 = VAE(repeat=repeat, channels=channels, dim_x=dim_x, dim_y=dim_y)
        net = VAEConcatCritic([pnet, qnet1, qnet2])
        net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    if critic_type != 'vae':
        num_epochs = 2
    else:
        num_epochs = 10 * repeat

    losses = []
    buffer = None

    for i in range(num_epochs):
        for j, x in enumerate(loaders):
            optimizer.zero_grad()
            bs = x[0].size(0)
            if bs != batch_size * repeat:
                pass
            x = torch.chunk(x[0], repeat, dim=0)
            # x = [v[0] for v in x]
            x, y = generate_test(
                x, imgs, masks, dim=dim_x, transform=transform)

            f = net(x, y).to(device)

            if critic_type == 'infonce':
                loss = -utils.infonce_lower_bound(f)
            elif critic_type == 'dv':
                loss = -utils.dv_upper_lower_bound(f)
            elif critic_type == 'nwj':
                loss = -utils.nwj_lower_bound(f)
            elif critic_type == 'reg_dv':
                loss = -utils.regularized_dv_bound(f, **kwargs)
            elif critic_type == 'smile':
                loss = -utils.smile_lower_bound(f, **kwargs)
            elif critic_type == 'mine':
                loss, buffer = utils.mine_lower_bound(f, buffer=buffer, momentum=0.9)
                loss = -loss
            elif critic_type == 'vae':
                loss = utils.vae_lower_bound(f)
                loss = -loss

                
            loss.backward()

            if debug and j % 20 == 0:
                if critic_type == 'vae':
                    print(f[0].mean().item(), f[1].mean().item())

                print(f'{j}: {-loss.item():.3f}')

            optimizer.step()

            mi = -loss.item()
            losses.append(mi)

    if dataset_name == 'mnist':
        dataset = datasets.MNIST('/atlas/u/tsong/data/mnist', train=False,
                                 transform=transforms.Compose([transforms.ToTensor()]))
    else:
        dataset = datasets.CIFAR10('/atlas/u/tsong/data/cifar', train=False,
                                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    test_loaders = td.DataLoader(dataset, batch_size=batch_size * repeat, shuffle=True,
                                 num_workers=2, pin_memory=True)

    with torch.no_grad():
        test_losses = []
        for j, x in enumerate(test_loaders):
            bs = x[0].size(0)
            if bs != batch_size * repeat:
                pass
            x = torch.chunk(x[0], repeat, dim=0)
            # x = [v[0] for v in x]
            x, y = generate_test(
                x, imgs, masks, dim=dim_x, transform=transform)
            # x = (x >= 0.5).float()
            # y = (y >= 0.5).float()

            f = net(x, y)
            # import ipdb; ipdb.set_trace()

            if critic_type == 'infonce':
                loss = -utils.infonce_lower_bound(f)
            elif critic_type == 'dv':
                loss = -utils.dv_upper_lower_bound(f)
            elif critic_type == 'nwj':
                loss = -utils.nwj_lower_bound(f)
            elif critic_type == 'reg_dv':
                loss = -utils.regularized_dv_bound(f, **kwargs)
            elif critic_type == 'smile':
                loss = -utils.smile_lower_bound(f, **kwargs)
            elif critic_type == 'mine':
                loss, buffer = utils.mine_lower_bound(
                    f, buffer=buffer, momentum=0.9)
                # print(loss.item())
                loss = -loss
            elif critic_type == 'vae':
                loss = utils.vae_lower_bound(f)
                loss = -loss

            if debug and j % 20 == 0:
                if critic_type == 'vae':
                    print(f[0].mean().item(), f[1].mean().item())

                print(f'{j}: {-loss.item():.3f}')

            mi = -loss.item()
            test_losses.append(mi)
    return np.array(losses), np.array(test_losses)


def cmdline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic', type=str, default='smile')
    parser.add_argument('--imgs', type=str, default='')
    parser.add_argument('--masks', type=str, default='')
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--transform', type=str, default='mask')

    args = parser.parse_args()

    if len(args.imgs) == 0:
        args.imgs = None
    else:
        args.imgs = eval(args.imgs)
    if len(args.masks) == 0:
        args.masks = None
    else:
        args.masks = eval(args.masks)

    k = args.critic
    critic_type = k.split('_')[0]

    if len(k.split('_')) > 1:
        clip = float(k.split('_')[-1])
    else:
        clip = None

    losses, test_losses = image_mi_estimator(dataset_name=args.dataset, critic_type=critic_type, clip=clip,
                                             imgs=args.imgs, masks=args.masks, debug=True, test=args.test, transform=args.transform)
    logdir_t = f'/atlas/u/tsong/exps/mi/{args.dataset}/{args.exp}t'
    logdir = f'/atlas/u/tsong/exps/mi/{args.dataset}/{args.exp}'
    import os
    try:
        os.makedirs(logdir)
        os.makedirs(logdir_t)
    except:
        pass
    savedir = os.path.join(logdir, f'{args.critic}_{args.imgs}_{args.masks}')
    np.save(savedir, losses)
    savedir = os.path.join(logdir_t, f'{args.critic}_{args.imgs}_{args.masks}')
    np.save(savedir, test_losses)


if __name__ == '__main__':
    cmdline()
    # image_mi_estimator(dataset_ratio=1.0, critic_type='smile',
    #                    imgs=([0, 1], [0, 1]), masks=([0, 0], [24, 24]), debug=True)
    # image_mi_estimator(dataset_ratio=1.0, critic_type='smile', mask=28, repeat=2, switch=True)
    # image_mi_estimator(dataset_ratio=1.0, critic_type='smile', repeat=2, masks=(20, 23))
