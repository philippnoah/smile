from turtle import forward
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F

class ExpNet(nn.Module):
    def __init__(self, config):
        super(ExpNet, self).__init__()
        x_dim, y_dim = config.dims[:2]
        # self.conv1 = nn.Conv2d(2 * config.repeat * config.channels, 64, kernel_size=5, stride=2, padding=2)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        # self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(2 * x_dim * y_dim, 1024)
        self.layers = nn.Sequential(*[nn.Linear(1024, 1024), nn.ReLU()]*4)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        # x = F.relu(self.conv1(x), 2)
        # x = F.relu(self.conv2_drop(self.conv2(x)), 2)
        # x = x.view(-1, 320)
        x = self.flatten(x)
        x = self.fc0(x)
        x = self.layers(x)
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

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
        # breakpoint()
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.fc(z)
        return z

class VAE(nn.Module):
    def __init__(self, config, repeat=None):
        super(VAE, self).__init__()
        self.config = config
        x_dim, y_dim, z_dim = self.x_dim, self.y_dim, self.z_dim = config.dims
        channels = config.channels
        repeat = self.repeat = repeat or config.repeat

        self.conv1 = nn.Sequential(
            nn.Conv2d(repeat * channels, 64 * repeat, 5, 2, 2),
            nn.ReLU()
        )
        dim_x, dim_y = utils.next_conv_size(x_dim, y_dim, 5, 2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * repeat, 128 * repeat, 5, 2, 2),
            nn.ReLU()
        )
        self.dim_x, self.dim_y = utils.next_conv_size(dim_x, dim_y, 5, 2, 2)
        self.top_conv_x, self.top_conv_y = dim_x, dim_y
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * repeat * dim_x * dim_y, 1024 * repeat),
            nn.ReLU(),
            nn.Linear(1024 * repeat, z_dim * 2 * repeat)
        )
        self.enc = nn.Sequential(self.conv1, self.conv2, self.fc)

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
        xy_tiled = torch.cat((x_tiled, y_tiled), dim=2)
        size = [batch_size * batch_size, x.size(1) * 2] + list(x.shape[2:])
        xy_pairs = torch.reshape(xy_tiled, size)
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
