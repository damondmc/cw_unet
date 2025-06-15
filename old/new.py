import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SQRT2 = 1.4142135623730951

def default_init(scale=1.):
    """Adapted from JAX default_init for PyTorch."""
    scale = 1e-10 if scale == 0 else scale
    return nn.init.normal_ if scale == 1. else lambda t: nn.init.normal_(t, std=np.sqrt(scale / t.shape[0]))

class ResnetBlockBigGANpp(nn.Module):
    """ResBlock adapted from BigGAN for PyTorch, modified for 2D data."""
    def __init__(self, in_ch, out_ch=None, up=False, down=False, dropout=0.1, 
                 fir=False, fir_kernel=(1, 3, 3, 1), skip_rescale=True, 
                 init_scale=0., up_down_factor=2, act=nn.LeakyReLU(0.2)):
        super(ResnetBlockBigGANpp, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch is not None else in_ch
        self.up = up
        self.down = down
        self.dropout = dropout
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale
        self.init_scale = init_scale
        self.up_down_factor = up_down_factor
        self.act = act

        # GroupNorm: min(32, channels // 4)
        self.norm1 = nn.GroupNorm(num_groups=min(self.in_ch // 4, 32), num_channels=self.in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(self.out_ch // 4, 32), num_channels=self.out_ch, eps=1e-6)
        self.dropout_layer = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1)

        # Initialize weights
        nn.init._no_grad_fill_(self.conv1.weight, 0.)
        nn.init._no_grad_fill_(self.conv2.weight, 0.)
        default_init(self.init_scale)(self.conv1.weight)
        default_init(self.init_scale)(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

        # Shortcut conv if channels or resolution change
        if self.in_ch != self.out_ch or self.up or self.down:
            self.shortcut = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1)
            nn.init._no_grad_fill_(self.shortcut.weight, 0.)
            default_init()(self.shortcut.weight)
            nn.init.zeros_(self.shortcut.bias)
        else:
            self.shortcut = nn.Identity()

    def naive_upsample_2d(self, x, factor=2):
        return F.interpolate(x, scale_factor=factor, mode='nearest')

    def naive_downsample_2d(self, x, factor=2):
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)

    def forward(self, x, temb=None, train=True):
        h = self.act(self.norm1(x))

        # Up or down sampling
        if self.up:
            if self.fir:
                raise NotImplementedError("FIR upsampling not implemented; use fir=False")
            h = self.naive_upsample_2d(h, factor=self.up_down_factor)
            x = self.naive_upsample_2d(x, factor=self.up_down_factor)
        elif self.down:
            if self.fir:
                raise NotImplementedError("FIR downsampling not implemented; use fir=False")
            h = self.naive_downsample_2d(h, factor=self.up_down_factor)
            x = self.naive_downsample_2d(x, factor=self.up_down_factor)

        h = self.conv1(h)
        # Time embedding (optional, skipped if temb is None)
        if temb is not None:
            h += self.act(temb).view(-1, self.out_ch, 1, 1)

        h = self.act(self.norm2(h))
        h = self.dropout_layer(h) if train else h
        h = self.conv2(h)
        x = self.shortcut(x)

        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / SQRT2

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, size_filter_in=16, dropout_prob=0):
        super(UNet, self).__init__()
        self.size_filter_in = size_filter_in

        # Initialize filters
        kernel_init = nn.init.kaiming_normal_

        # Encoder
        self.encoder = nn.ModuleList([
            self.conv_block(input_channels, size_filter_in, kernel_init),
            self.conv_block(size_filter_in, size_filter_in * 2, kernel_init),
            self.conv_block(size_filter_in * 2, size_filter_in * 4, kernel_init),
            self.conv_block(size_filter_in * 4, size_filter_in * 8, kernel_init)
        ])

        # Downsampling with ResnetBlockBigGANpp
        self.downsample = nn.ModuleList([
            ResnetBlockBigGANpp(size_filter_in, size_filter_in, down=True, dropout=dropout_prob),
            ResnetBlockBigGANpp(size_filter_in * 2, size_filter_in * 2, down=True, dropout=dropout_prob),
            ResnetBlockBigGANpp(size_filter_in * 4, size_filter_in * 4, down=True, dropout=dropout_prob),
            ResnetBlockBigGANpp(size_filter_in * 8, size_filter_in * 8, down=True, dropout=dropout_prob)
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            self.conv_block(size_filter_in * 8, size_filter_in * 16, kernel_init),
            nn.Dropout(dropout_prob)
        )

        # Decoder
        self.upsample = nn.ModuleList([
            ResnetBlockBigGANpp(size_filter_in * 16, size_filter_in * 8, up=True, dropout=dropout_prob),
            ResnetBlockBigGANpp(size_filter_in * 8, size_filter_in * 4, up=True, dropout=dropout_prob),
            ResnetBlockBigGANpp(size_filter_in * 4, size_filter_in * 2, up=True, dropout=dropout_prob),
            ResnetBlockBigGANpp(size_filter_in * 2, size_filter_in, up=True, dropout=dropout_prob)
        ])

        self.decoder = nn.ModuleList([
            self.conv_block(size_filter_in * 16, size_filter_in * 8, kernel_init),
            self.conv_block(size_filter_in * 8, size_filter_in * 4, kernel_init),
            self.conv_block(size_filter_in * 4, size_filter_in * 2, kernel_init),
            self.conv_block(size_filter_in * 2, size_filter_in, kernel_init)
        ])

        # Output layer
        self.output_layer = nn.Conv2d(size_filter_in, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_init):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        for m in block:
            if isinstance(m, nn.Conv2d):
                kernel_init(m.weight)
                nn.init.zeros_(m.bias)
        return block

    def forward(self, x):
        # Encoder
        encoder_memory = []
        for enc, down in zip(self.encoder, self.downsample[:-1]):  # Last downsample applied after loop
            x = enc(x)
            encoder_memory.append(x)
            x = down(x, train=True)

        # Last downsample before bottleneck
        x = self.downsample[-1](x, train=True)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, dec in zip(self.upsample, self.decoder):
            x = up(x, train=True)
            memory = encoder_memory.pop()
            # Center crop memory to match upsampled size if needed
            if x.shape[2:] != memory.shape[2:]:
                memory = F.center_crop(memory, [x.shape[2], x.shape[3]])
            x = torch.cat((x, memory), dim=1)
            x = dec(x)

        # Output
        x = self.output_layer(x)
        return torch.tanh(x)