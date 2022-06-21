import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import json
import matplotlib.pyplot as plt
import utils
from datetime import datetime as dt
from critics import VAE, VAEConcatCritic, ConcatCritic, ConvNet, ExpNet, SeparableCritic
from statistics import mean


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

def train(train_dataloader, net, config):
    if config.debug:
        print("Starting training ...")

    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    train_losses = []
    no_impr_count = 0

    for i in range(config.epochs):
        if not train_dataloader.dataset.load(i):
            break

        for j, x in enumerate(train_dataloader):
            if j > config.iterations:
                break
            optimizer.zero_grad()
            # x = torch.chunk(x, config.repeat, dim=0)
            x, y = generate_test_for_translation(x, config)
            # x, y = [x_.to(config.device) for x_ in x]
            x, y = x.to(config.device), y.to(config.device)
            f = net(x, y).to(config.device)
            loss = MI(f, config)
            loss.backward()

            optimizer.step()
            mi = -loss.item()
            if mi <= (train_losses or [-10])[-1]:
                no_impr_count += 1
                if no_impr_count >= config.early_stopping:
                    break
            else:
                no_impr_count = 0 

            train_losses.append(max(mi,-0.5))

            if (j+1) % 20 == 0 and config.debug:
                [print(f"{n:.3f}", end=", ") for n in train_losses[-20:]]
                print()
                print(f'{j+1}: {mean(train_losses[-20:]):.3f}')
                if config.estimator_name == 'vae':
                    print(f[0].mean().item(), f[1].mean().item())
            
        if no_impr_count >= config.early_stopping:
            print("Early stopping ....")
            break
        
        if config.debug:
            plot(np.array(train_losses), config, "train")

    if config.debug:
        print("Done.")

    return np.array(train_losses)

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

def generate_test_for_translation(X, config):
    x, y_ = X
    l = int(y_.size(2) * config.keep)
    y = torch.zeros_like(y_)
    y[:,:,:l] = y_[:,:,:l]
    return x, y

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
        loss = -utils.smile_lower_bound(f, alpha=config.alpha, clip=config.clip, device=config.device)
    elif config.estimator_name == 'mine':
        loss, config.buffer = utils.mine_lower_bound(f, buffer=config.buffer, momentum=0.9, device=config.device)
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

    if config.debug:
        print(f"Keeping {int(config.keep * 512)} embeddings")

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
    
    # config.filename = (
    #     f"{config.estimator_name}_"
    #     f"{str(config.imgs).replace(' ', '')}_"
    #     f"{str(config.masks).replace(' ', '')}"
    #     )

    tim = dt.now().strftime('%Y-%m-%d-_%H:%M:%S')
    config.filename = f"{tim}"

    losses_dir = "losses"
    config.losses_path = os.path.join(losses_dir, config.filename)
    os.makedirs(losses_dir, exist_ok=True)

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
    parser.add_argument('--critic-net-name', type=str, default='conv', choices=['conv', 'vae', 'exp', 'sep'])
    parser.add_argument('--imgs', type=str, default=None)
    parser.add_argument('--masks', type=str, default=None)
    parser.add_argument('--keep', type=float, default=1.0)
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dataset-name', type=str, default='mnist', choices=['mnist', 'cifar', 'text', 'bert', 'text', 'text/translation', 'text/translation/random'])
    parser.add_argument('--transform', type=str, default='mask')
    parser.add_argument('--dataset-ratio', type=float, default=1.0)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--iterations', type=float, default=100)#float("inf"))
    parser.add_argument('--fig-dir', type=str, default='figs/')
    parser.add_argument('--num-rows', type=int, default=None)
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--num-layers', type=int, default=0)
    parser.add_argument('--embed-dim', type=int, default=1)
    parser.add_argument('--early-stopping', type=int, default=1000, help="How many batches with consecutively increasing losses should be allowed")
    args = parser.parse_args()
    return args

def load_dataset(config):
    transf = transforms.ToTensor()
    dataset = None
    if config.dataset_name.startswith('text'):
        seq = "seq" in config.dataset_name
        random = "random" in config.dataset_name
        translation = "translation" in config.dataset_name
        dataset = utils.TranslationDataset(token_limit=None, seq=seq, random=random, translation=translation, keep=config.keep)
        config.channels = dataset.EN.shape[1]
        config.dims = dataset.EN.shape[2:]
    elif config.dataset_name == 'bert':
        file_name = 'data/text/bert_embeddings_10.pt'
        dataset = utils.TextDataset(file_name, num_rows=config.num_rows, model_name=config.dataset_name)
        config.channels = dataset.x_train.shape[1]
        config.dims = dataset.x_train.shape[2:]
    elif config.dataset_name == 'old_text':
        file_name = 'data/text/length_10.npy'
        dataset = utils.TextDataset(file_name, num_rows=config.num_rows, model_name=config.dataset_name)
        config.channels = dataset.x_train.shape[1]
        config.dims = dataset.x_train.shape[2:]
        dataset.x_train = dataset.x_train[:,:,:config.dims[0],:]
    elif config.dataset_name == 'mnist':
        dataset = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([transf]))
        config.channels = 1
        config.dims = [28, 28]
    elif config.dataset_name == 'cifar':
        normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        dataset = datasets.CIFAR10('data/cifar/', train=True, download=True,\
         transform=transforms.Compose([transf, normal]))
        config.channels = 3
        config.dims = [32, 32]

    if config.estimator_name == 'vae':
        config.dims += [10] # add z dim

    # random_indices = np.random.permutation(np.arange(0, int(len(dataset) * config.dataset_ratio)))

    # dataset = td.Subset(dataset, random_indices)

    train_dataloader = td.DataLoader(dataset, batch_size=config.batch_size * config.repeat, 
        shuffle=False, num_workers=2, pin_memory=True)
        
    test_dataloader = td.DataLoader(dataset, batch_size=config.batch_size * config.repeat, 
        shuffle=False, num_workers=2, pin_memory=True)

    return train_dataloader, test_dataloader

def init_estimator(config):
    if config.critic_net_name == 'exp':
        config.channels = 1
        net = ExpNet(config)
        net = ConcatCritic(net)
        net.to(config.device)
    elif config.critic_net_name == 'sep':
        net = SeparableCritic(config)
        net.to(config.device)
    elif config.critic_net_name == 'vae':
        pnet = VAE(config, repeat=config.repeat * 2)
        qnet1 = VAE(config)
        qnet2 = VAE(config)
        net = VAEConcatCritic([pnet, qnet1, qnet2])
        net.to(config.device)
    elif config.critic_net_name == 'conv':
        config.channels = 1
        conv_net = ConvNet(config)
        net = ConcatCritic(conv_net)
        net.to(config.device)
    return net

def plot(losses: np.array, config, label):

    def moving_average(interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')[:-window_size]


    # plt.plot(list(range(len(losses))), losses, label=label, marker=',', linestyle='-', color='lightgrey', zorder=1)
    plt.scatter(np.array(range(0, len(losses))), losses, label=label, marker='.', zorder=2)
    line = moving_average(losses, 20)
    plt.plot(line, color='red')
    plt.title(f"dataset={config.dataset_name} -- keep={config.keep} -- estimator={config.critic_net_name}")
    # plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('MI')
    path = f"{config.fig_file_path}_{str(config.keep).replace('.', '')}_{label}"
    plt.savefig(path)
    plt.clf()

    description_file = open(f"{config.fig_file_path}_params.json", "w")
    polished_config = copy.deepcopy(vars(config))
    polished_config.pop('device', None)
    polished_config.pop('buffer', None)
    json.dump(polished_config, description_file, indent=4, sort_keys=True)

    np.save(config.losses_path, losses)

def save_losses(losses, test_losses, config):
    if config.debug:
        print("Saving losses ... ")

    np.save(config.losses_path, losses)

    if config.debug:
        print("Done.")

if __name__ == '__main__':

    # initialize
    config = init_config()
    train_dataloader, test_dataloader = load_dataset(config)
    net = init_estimator(config)

    # train, test
    # breakpoint()
    train_losses = train(train_dataloader, net, config)
    plot(train_losses, config, label='train')

    exit()

    test_losses = test(test_dataloader, net, config)
    plot(test_losses, config, label='test')

    # save results
    save_losses(train_losses, test_losses, config)
