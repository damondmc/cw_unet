import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, size_filter_in=16, dropout_prob=0):
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
            nn.Dropout(dropout_prob)
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
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
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
        for (up_sample, layer) in zip(self.up_sample_layer, self.decoder):
            x = up_sample(x)
            x = torch.cat((x, encoder_memory.pop()), dim=1) # list.pop() remove and reurn the last element in the list
            x = layer(x)
        # Output
        x = self.output_layer(x)
        return torch.tanh(x)
        #return x
    
