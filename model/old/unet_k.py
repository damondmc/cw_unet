import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


"""
Source: https://github.com/LeeJunHyun/Image_Segmentation.git
"""

def init_weights(net, init_type='normal', gain=0.02):
    """Initialize network weights using specified initialization method.
    
    Args:
        net (nn.Module): The neural network to initialize.
        init_type (str): Type of initialization ('normal', 'xavier', 'kaiming', 'orthogonal').
        gain (float): Scaling factor for initialization (default: 0.02).
    
    Raises:
        NotImplementedError: If the specified init_type is not supported.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    """A double convolutional block with two Conv2d-BatchNorm2d-ReLU sequences.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through the double convolutional block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, in_channels, height, width].
        
        Returns:
            torch.Tensor: Output tensor after two Conv-BN-ReLU sequences.
        """
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """Upsampling block with interpolation, convolution, batch norm, and ReLU.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # Doubles spatial dimensions
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through the upsampling block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, in_channels, height, width].
        
        Returns:
            torch.Tensor: Output tensor of shape [batch, out_channels, 2*height, 2*width].
        """
        x = self.up(x)
        return x
        
class Attention_block(nn.Module):
    """Attention mechanism to weight encoder features based on decoder input.
    
    Args:
        F_g (int): Number of channels in the gating signal (decoder).
        F_l (int): Number of channels in the encoder feature map.
        F_int (int): Number of intermediate channels for attention computation.
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.LeakyReLU()

    def forward(self, g, x):
        """Forward pass to compute attention-weighted encoder features.
        
        Args:
            g (torch.Tensor): Gating signal from decoder [batch, F_g, height, width].
            x (torch.Tensor): Encoder feature map [batch, F_l, height, width].
        
        Returns:
            torch.Tensor: Attention-weighted encoder features [batch, F_l, height, width].
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # Apply attention weights to encoder features
    
class Attention_UNet(nn.Module):
    """Attention U-Net for image segmentation with encoder and decoder in nn.ModuleList.
    
    Args:
        in_channels (int): Number of input image channels (default: 3).
        out_channels (int): Number of output channels (default: 1).
        latent_channels (int): Number of channels in the first encoder layer (default: 64).
    """
    def __init__(self, in_channels=3, out_channels=1, latent_channels=64, dropout_prob=0):
        super(Attention_UNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        # Encoder packed into nn.ModuleList
        # Each tuple contains (conv_block, Maxpool) for each encoder level
        self.encoder = nn.ModuleList([
            conv_block(in_channels=in_channels, out_channels=latent_channels),
            conv_block(in_channels=latent_channels, out_channels=latent_channels*2),
            conv_block(in_channels=latent_channels*2, out_channels=latent_channels*4),
            conv_block(in_channels=latent_channels*4, out_channels=latent_channels*8)
        ])

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            conv_block(in_channels=latent_channels*8, out_channels=latent_channels*16),
            nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        )
        
        # Decoder packed into nn.ModuleList
        # Each tuple contains (up_conv, Attention_block, conv_block, in_channels for concatenation)
        self.decoder = nn.ModuleList([
            conv_block(in_channels=latent_channels*16, out_channels=latent_channels*8),
            conv_block(in_channels=latent_channels*8, out_channels=latent_channels*4),
            conv_block(in_channels=latent_channels*4, out_channels=latent_channels*2),
            conv_block(in_channels=latent_channels*2, out_channels=latent_channels)
          
        ])
        
        self.upsample_layer = nn.ModuleList([
            up_conv(in_channels=latent_channels*16, out_channels=latent_channels*8),
            up_conv(in_channels=latent_channels*8, out_channels=latent_channels*4),
            up_conv(in_channels=latent_channels*4, out_channels=latent_channels*2),
            up_conv(in_channels=latent_channels*2, out_channels=latent_channels)
        ])
        
        
        self.attention_layer = nn.ModuleList([
            Attention_block(F_g=latent_channels*8, F_l=latent_channels*8, F_int=latent_channels*4),
            Attention_block(F_g=latent_channels*4, F_l=latent_channels*4, F_int=latent_channels*2),
            Attention_block(F_g=latent_channels*2, F_l=latent_channels*2, F_int=latent_channels),
            Attention_block(F_g=latent_channels, F_l=latent_channels, F_int=latent_channels//2)
        ])

        self.output_layer = nn.Sequential(
            nn.Conv2d(latent_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        """Forward pass through Attention U-Net with encoder-decoder and attention gates.
        
        Args:
            x (torch.Tensor): Input image tensor [batch, in_channels, height, width].
        
        Returns:
            torch.Tensor: Segmentation output [batch, out_channels, height, width].
        """
        # Encoder path using nn.ModuleList
        skip_connections = []
        for i, conv in enumerate(self.encoder):
            x = conv(x)
            skip_connections.append(x)  # Store before Maxpool
            x = self.Maxpool(x)

        # Bottleneck layer
        x = self.bottleneck(x)

        # Decoder path using nn.ModuleList
        for i, (up, att, conv) in enumerate(zip(self.upsample_layer, self.attention_layer, self.decoder)):
            x = up(x)  # Upsample
            # Apply attention to encoder skip connection
            ag_x = att(g=x, x=skip_connections[-(i+1)])  # Reverse order: index -1 for x4, -2 for x3, etc.
            # Concatenate with upsampled features
            x = torch.cat((ag_x, x), dim=1)
            x = conv(x)  # Apply conv block

        # Final 1x1 convolution for output
        x = self.output_layer(x)
        return x           
