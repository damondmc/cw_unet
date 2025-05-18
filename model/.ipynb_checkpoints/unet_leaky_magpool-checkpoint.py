import torch
import torch.nn as nn
import torch.nn.functional as F

def MagPool2d(data, kernel_size, stride=None, padding=0):
    """
    Magnitude Pooling based on magnitude using two pooling operations.
    
    Args:
        data (torch.Tensor): Input tensor of shape (k, c, m, n).
        kernel_size (int or tuple): Size of the pooling window.
        stride (int or tuple, optional): Stride of the pooling. Defaults to kernel_size.
        padding (int or tuple, optional): Padding to be added before pooling. Defaults to 0.
        
    Returns:
        torch.Tensor: Downsampled tensor, with the same sign as the original values.
    """
    # Perform max pooling on the original tensor and its negation
    pos_pool = F.max_pool2d(data, kernel_size, stride, padding)
    neg_pool = F.max_pool2d(-data, kernel_size, stride, padding)
    
    # Combine results: select the value with the larger magnitude
    return torch.where(pos_pool >= neg_pool, pos_pool, -neg_pool)




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

        self.up_sample_layer = nn.ModuleList([
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
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
    
    def CropAndConcat(self, x, memory):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim=1)
        return x

    def down_sample(self, x):
        return nn.MagPool2d(x, kernel_size=2)
    
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
        for (up_sample, layer) in zip(self.up_sample_layer, self.decoder):
            x = up_sample(x)
            x = torch.cat((x, encoder_memory.pop()), dim=1) # list.pop() remove and reurn the last element in the list
            x = layer(x)
        # Output
        x = self.output_layer(x)
        #return torch.tanh(x)
        return x
    