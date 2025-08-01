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

class single_conv(nn.Module):
    """A single convolutional block with Conv2d, BatchNorm2d, and ReLU.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)  # In-place ReLU to save memory
        )

    def forward(self, x):
        """Forward pass through the convolutional block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, in_channels, height, width].
        
        Returns:
            torch.Tensor: Output tensor after convolution, batch norm, and ReLU.
        """
        x = self.conv(x)
        return x

class conv_block(nn.Module):
    """A double convolutional block with two Conv2d-BatchNorm2d-ReLU sequences.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

class Recurrent_block(nn.Module):
    """Recurrent convolutional block for feature refinement with repeated convolutions.
    
    Args:
        out_channels (int): Number of output channels.
        t (int): Number of recurrent iterations (default: 2).
    """
    def __init__(self, out_channels, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass with recurrent convolution, adding input to each iteration.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, out_channels, height, width].
        
        Returns:
            torch.Tensor: Output tensor after t recurrent convolutions.
        """
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)  # Add input to recurrent output
        return x1
        
class RRCNN_block(nn.Module):
    """Recurrent Residual Convolutional Neural Network block with two recurrent blocks.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        t (int): Number of recurrent iterations (default: 2).
    """
    def __init__(self, in_channels, out_channels, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(out_channels, t=t),
            Recurrent_block(out_channels, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward pass with 1x1 convolution and recurrent blocks, adding residual connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, in_channels, height, width].
        
        Returns:
            torch.Tensor: Output tensor of shape [batch, out_channels, height, width].
        """
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1  # Residual connection

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
        
        self.relu = nn.ReLU(inplace=True)

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
    
class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNet, self).__init__()

        # Initialize filters and kernel weights
        size_filter_in = 16
        kernel_init = nn.init.kaiming_normal_
        # Encoder
        self.encoder = nn.ModuleList([
            self.conv_block(input_channels, size_filter_in, kernel_init),
            self.conv_block(size_filter_in, size_filter_in * 2, kernel_init),
            self.conv_block(size_filter_in * 2, size_filter_in * 4, kernel_init),
            self.conv_block(size_filter_in * 4, size_filter_in * 8, kernel_init)
        ])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            self.conv_block(size_filter_in * 8, size_filter_in * 16, kernel_init),
        #    nn.Dropout(0.5)
        )
        
        # Decoder
        self.decoder = nn.ModuleList([
            self.conv_block(size_filter_in * 16, size_filter_in * 8, kernel_init),
            self.conv_block(size_filter_in * 8, size_filter_in * 4, kernel_init),
            self.conv_block(size_filter_in * 4, size_filter_in * 2, kernel_init),
            self.conv_block(size_filter_in * 2, size_filter_in, kernel_init)
        ])

        self.upsample_layer = nn.ModuleList([
            nn.ConvTranspose2d(size_filter_in * 16, size_filter_in * 8, kernel_size=2, stride=2),
            nn.ConvTranspose2d(size_filter_in * 8, size_filter_in * 4, kernel_size=2, stride=2),
            nn.ConvTranspose2d(size_filter_in * 4, size_filter_in * 2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(size_filter_in * 2, size_filter_in, kernel_size=2, stride=2)
        ])
        # Output layer
        self.output_layer = nn.Conv2d(size_filter_in, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_init):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def CropAndConcat(self, x, memory):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim=1)
        return x

    def down_sample(self, x):
        return nn.MaxPool2d(2)(x)
    
    def forward(self, x):
        # Encoder
        encoder_memory = []
        for layer in self.encoder:
            x = layer(x)
            encoder_memory.append(x)
            x = self.down_sample(x)
            
        # Bottleneck
        x = self.bottleneck(x)
    
        # Decoder
        for (up_sample, layer) in zip(self.upsample_layer, self.decoder):
            x = up_sample(x)
            x = torch.cat((x, encoder_memory.pop()), dim=1) # list.pop() remove and reurn the last element in the list
            x = layer(x)
        # Output
        x = self.output_layer(x)
        #return torch.tanh(x)
        return x

class R2_UNet(nn.Module):
    """Recurrent Residual U-Net (R2U-Net) for image segmentation with encoder and decoder in nn.ModuleList.
    
    Args:
        in_channels (int): Number of input image channels (default: 4).
        out_channels (int): Number of output channels (default: 4).
        t (int): Number of recurrent iterations in RRCNN blocks (default: 2).
        latent_channels (int): Number of channels in the first encoder layer (default: 64).
        dropout_prob (float): Dropout probability for regularization (default: 0).
    """
    def __init__(self, in_channels=4, out_channels=4, t=2, latent_channels=64, dropout_prob=0):
        super(R2_UNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        # Encoder packed into nn.ModuleList (only RRCNN_block instances)
        self.encoder = nn.ModuleList([
            RRCNN_block(in_channels=in_channels, out_channels=latent_channels, t=t),
            RRCNN_block(in_channels=latent_channels, out_channels=latent_channels*2, t=t),
            RRCNN_block(in_channels=latent_channels*2, out_channels=latent_channels*4, t=t),
            RRCNN_block(in_channels=latent_channels*4, out_channels=latent_channels*8, t=t)
        ])

        # Bottleneck layer with dropout
        self.bottleneck = nn.Sequential(
            RRCNN_block(in_channels=latent_channels*8, out_channels=latent_channels*16, t=t),
            nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        )

        # Decoder packed into nn.ModuleList
        # Each tuple contains (up_conv, RRCNN_block, in_channels for concatenation)
        self.decoder = nn.ModuleList([
            RRCNN_block(in_channels=latent_channels*16, out_channels=latent_channels*8, t=t),
            RRCNN_block(in_channels=latent_channels*8, out_channels=latent_channels*4, t=t),
            RRCNN_block(in_channels=latent_channels*4, out_channels=latent_channels*2, t=t),
            RRCNN_block(in_channels=latent_channels*2, out_channels=latent_channels, t=t)
        
        ])
        
        self.upsample_layer = nn.ModuleList([
            up_conv(in_channels=latent_channels*16, out_channels=latent_channels*8),
            up_conv(in_channels=latent_channels*8, out_channels=latent_channels*4),
            up_conv(in_channels=latent_channels*4, out_channels=latent_channels*2),
            up_conv(in_channels=latent_channels*2, out_channels=latent_channels)
        ])

        self.Conv_1x1 = nn.Conv2d(latent_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward pass through R2U-Net with encoder-decoder and skip connections.
        
        Args:
            x (torch.Tensor): Input image tensor [batch, in_channels, height, width].
        
        Returns:
            torch.Tensor: Segmentation output [batch, out_channels, height, width].
        """
        # Encoder path using nn.ModuleList
        skip_connections = []
        for i, rrcnn in enumerate(self.encoder):
            x = rrcnn(x)
            skip_connections.append(x)
            x = self.Maxpool(x)  # Apply max pooling after storing skip connection

        # Bottleneck layer
        x = self.bottleneck(x)

        # Decoder path using nn.ModuleList
        for i, (up, rrcnn) in enumerate(zip(self.upsample_layer, self.decoder)):
            x = up(x)  # Upsample
            # Concatenate with corresponding encoder skip connection (in reverse order)
            x = torch.cat((skip_connections[-(i+1)], x), dim=1)
            x = rrcnn(x)  # Apply RRCNN block

        # Final 1x1 convolution for output
        x = self.Conv_1x1(x)
        return x

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

        self.Conv_1x1 = nn.Conv2d(latent_channels, out_channels, kernel_size=1, stride=1, padding=0)

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
        x = self.Conv_1x1(x)
        return x

class R2Att_UNet(nn.Module):
    """Attention R2U-Net combining recurrent residual blocks and attention gates with encoder and decoder in nn.ModuleList.
    
    Args:
        in_channels (int): Number of input image channels (default: 3).
        out_channels (int): Number of output channels (default: 1).
        t (int): Number of recurrent iterations in RRCNN blocks (default: 2).
        latent_channels (int): Number of channels in the first encoder layer (default: 64).
    """
    def __init__(self, in_channels=3, out_channels=1, t=2, latent_channels=64, dropout_prob=0):
        super(R2Att_UNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        # Encoder packed into nn.ModuleList
        # Each tuple contains (RRCNN_block, Maxpool) for each encoder level
        self.encoder = nn.ModuleList([
            RRCNN_block(in_channels=in_channels, out_channels=latent_channels, t=t),
            RRCNN_block(in_channels=latent_channels, out_channels=latent_channels*2, t=t),
            RRCNN_block(in_channels=latent_channels*2, out_channels=latent_channels*4, t=t),
            RRCNN_block(in_channels=latent_channels*4, out_channels=latent_channels*8, t=t)
        ])

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            RRCNN_block(in_channels=latent_channels*8, out_channels=latent_channels*16, t=t),
            nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        )

        # Decoder packed into nn.ModuleList
        # Each tuple contains (up_conv, Attention_block, RRCNN_block, in_channels for concatenation)
        self.decoder = nn.ModuleList([
            RRCNN_block(in_channels=latent_channels*16, out_channels=latent_channels*8, t=t),
            RRCNN_block(in_channels=latent_channels*8, out_channels=latent_channels*4, t=t),
            RRCNN_block(in_channels=latent_channels*4, out_channels=latent_channels*2, t=t),
            RRCNN_block(in_channels=latent_channels*2, out_channels=latent_channels, t=t)
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

        self.Conv_1x1 = nn.Conv2d(latent_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward pass through Attention R2U-Net with encoder-decoder, attention, and skip connections.
        
        Args:
            x (torch.Tensor): Input image tensor [batch, in_channels, height, width].
        
        Returns:
            torch.Tensor: Segmentation output [batch, out_channels, height, width].
        """
        # Encoder path using nn.ModuleList
        skip_connections = []
        for i, rrcnn in enumerate(self.encoder):
            x = rrcnn(x)
            skip_connections.append(x)  # Store before Maxpool
            x = self.Maxpool(x)

        # Bottleneck layer
        x = self.bottleneck(x)

        # Decoder path using nn.ModuleList
        for i, (up, att, rrcnn) in enumerate(zip(self.upsample_layer, self.attention_layer, self.decoder)):
            x = up(x)  # Upsample
            # Apply attention to encoder skip connection
            ag_x = att(g=x, x=skip_connections[-(i+1)])  # Reverse order: index -1 for x4, -2 for x3, etc.
            # Concatenate with upsampled features
            x = torch.cat((ag_x, x), dim=1)
            x = rrcnn(x)  # Apply RRCNN block

        # Final 1x1 convolution for output
        x = self.Conv_1x1(x)
        return x