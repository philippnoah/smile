import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import utils
from datetime import datetime as dt

from critics import VAE, VAEConcatCritic, ConcatCritic, ConvNet, ExpNet

def train(train_dataloader, net, config):
    if config.debug:
        print("Starting training ...")

    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    train_losses = []

    for i in range(config.epochs):
        for j, (x, t) in enumerate(train_dataloader):
            if j > config.iterations:
                break
            optimizer.zero_grad()
            x = torch.chunk(x, config.repeat, dim=0)
            x, y = generate_test(x, config)
            f = net(x).to(config.device)
            loss = MI(f, config)
            loss.backward()

            if j % 20 == 0 and config.debug:
                print(f'{j}: {-loss.item():.3f}')
                if config.estimator_name == 'vae':
                    print(f[0].mean().item(), f[1].mean().item())

            optimizer.step()
            mi = -loss.item()
            train_losses.append(mi)

    if config.debug:
        print("Done.")

    return np.array(train_losses)

def test(test_dataloader, net, config):
    if config.debug:
        print("Starting testing ...")

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

    if config.debug:
        print("Done.")

    return np.array(test_losses)

def generate_mask(mask, config, batch_size):
    channels = config.channels
    x_dim, y_dim = config.dims[:2]
    return torch.cat([
        torch.ones(batch_size, channels, x_dim - mask, y_dim),
        torch.zeros(batch_size, channels, mask, y_dim)
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
    batch_size = X[0].shape[0]

    x = [X[i].to(config.device) for i in config.imgs[0]]
    y = [X[i].to(config.device) for i in config.imgs[1]]

    if config.transform == 'mask':
        mx = [generate_mask(m, config, batch_size) for m in config.masks[0]]
        my = [generate_mask(m, config, batch_size) for m in config.masks[1]]

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

def init_config():
    print("Initializing ...")

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

    if config.debug:
        print("Done.")

    return config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator-name', type=str, default='smile')
    parser.add_argument('--critic-net-name', type=str, default='conv', choices=['conv', 'vae', 'exp'])
    parser.add_argument('--imgs', type=str, default=None)
    parser.add_argument('--masks', type=str, default=None)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dataset-name', type=str, default='mnist')
    parser.add_argument('--transform', type=str, default='mask')
    parser.add_argument('--dataset-ratio', type=float, default=1.0)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--iterations', type=float, default=1000)#float("inf"))
    parser.add_argument('--fig-dir', type=str, default='figs/')
    args = parser.parse_args()
    return args

def load_dataset(config):
    transf = transforms.ToTensor()
    if config.dataset_name == 'text':
        dataset = utils.TextDataset('data/text/token_embeddings_sm.npy')
        config.channels = dataset.x_train[0].shape[0]
        config.dims = dataset.x_train[0].shape[1:]
        dataset.x_train = dataset.x_train[:,:,:config.dims[0],:]
    elif config.dataset_name == 'mnist':
        dataset = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([transf]))
        config.channels = 1
        config.dims = [28, 28]
    else:
        normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        dataset = datasets.CIFAR10('data/cifar/', train=True, download=True, \
            transform=transforms.Compose([transf, normal]))
        config.channels = 3
        config.dims = [32, 32]

    if config.estimator_name == 'vae':
        config.dims += [10] # add z dim

    random_indices = np.random.permutation(np.arange(0, int(len(dataset) * config.dataset_ratio)))

    dataset = td.Subset(dataset, random_indices)

    train_dataloader = td.DataLoader(dataset, batch_size=config.batch_size * config.repeat, \
        shuffle=True, num_workers=2, pin_memory=True)
        
    test_dataloader = td.DataLoader(dataset, batch_size=config.batch_size * config.repeat, \
        shuffle=True, num_workers=2, pin_memory=True)

    return train_dataloader, test_dataloader

def init_estimator(config):
    if config.critic_net_name == 'exp' or config.dataset_name == 'text':
        config.channels = 1
        net = ExpNet(config)
        # net = ConcatCritic(conv_net)
        net.to(config.device)
    elif config.critic_net_name == 'vae':
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
    if config.debug:
        print("Saving losses ... ")

    np.save(config.loss_log_path, losses)
    np.save(config.loss_log_path_t, test_losses)

    if config.debug:
        print("Done.")

if __name__ == '__main__':

    # initialize
    config = init_config()
    train_dataloader, test_dataloader = load_dataset(config)
    net = init_estimator(config)

    # train, test
    train_losses = train(train_dataloader, net, config)
    plot(train_losses, config, label='train')

    exit()

    test_losses = test(test_dataloader, net, config)
    plot(test_losses, config, label='test')

    # save results
    save_losses(train_losses, test_losses, config)