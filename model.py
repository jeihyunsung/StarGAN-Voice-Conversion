import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loader import get_loader


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        self.actv = nn.ReLU(inplace=True)

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        return x + self._residual(x, s)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, num_speakers=10, conv_dim=64, repeat_num=6):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        self.downsample = nn.Sequential(*layers)

        # Bottleneck layers.
        self.adain = nn.ModuleList()
        for i in range(repeat_num):
            self.adain.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        layers = []
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        self.upsample = nn.Sequential(*layers)

    def forward(self, x, s):
        # Replicate spatially and concatenate domain information.
        out = self.downsample(x)
        for block in self.adain:
            out = block(out, s)
        out = self.upsample(out)

        return out

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 256), conv_dim=64, repeat_num=5, num_speakers=10):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False) # padding should be 0
        self.conv_clf_spks = nn.Conv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False)  # for num_speaker
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_dis(h)
        out_cls_spks = self.conv_clf_spks(h)
        return out_src, out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        style = s.view([-1,s.size(2)])
        h = self.fc(style)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_speakers=10):
        super(MappingNetwork, self).__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 256)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(256, 256)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_speakers):
            self.unshared += [nn.Sequential(nn.Linear(256, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_speakers, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s

class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_speakers=10, max_conv_dim=256):
        super(StyleEncoder, self).__init__()
        dim_in = 64
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        # repeat_num = int(np.log2(img_size)) - 2
        repeat_num = 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in=dim_in, dim_out=dim_out, downsample=True)]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, kernel_size=(3, 9), stride=(1, 2), padding=0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, kernel_size=(3, 9), stride=(1, 2), padding=0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, kernel_size=(3, 9), stride=(1, 2), padding=0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, kernel_size=(3, 1), stride=1, padding=0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_speakers):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_speakers, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        s = s.unsqueeze_(1)
        return s

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_loader('/hdd_4T/hynsng/mc/train', 1, 'train', num_workers=1)
    data_iter = iter(train_loader)
    G = Generator().to(device)
    D = Discriminator().to(device)
    E = StyleEncoder().to(device)
    M = MappingNetwork().to(device)
    for i in range(10):
        mc_real, mc2, spk_label_org, spk_c_org = next(data_iter)
        mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
        mc_real = mc_real.to(device)                         # Input mc.
        spk_label_org = spk_label_org.to(device)             # Original spk labels.
        spk_c_org = spk_c_org.to(device)             # Original spk acc conditioning.

        style_code_mc = E(mc_real, spk_label_org)

        latent_dim = 16
        z_trg = torch.randn(16, latent_dim)
        z_trg = z_trg.unsqueeze_(1)
        z_trg = z_trg.to(device)
        style_code_lt = M(z_trg, spk_label_org)

        mc_real_mod = torch.cat([mc_real, mc_real, mc_real],dim=3)
        mc_real_mod2 = mc_real_mod[:,:,:,:672]
        print(mc_real_mod2.size())
        mc_fake = G(mc_real_mod2, style_code_lt)
        out_src, out_cls_spks = D(mc_fake)


