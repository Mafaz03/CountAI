import torch
from torch import nn
import matplotlib.pyplot as plt

# Define a convolutional block used in the discriminator
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        # Sequential block: Conv2d -> BatchNorm2d -> LeakyReLU
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=4, 
                stride=stride,
                padding=1,
                bias="False",  # NOTE: Should be False (not string), but kept as in original code
                padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        # Forward pass through the convolutional block
        return self.conv(x)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features = [64, 128, 256, 512]):
        super().__init__()

        # Initial convolutional layer (no batch norm)
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=features[0], 
                kernel_size=4, 
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_c = features[0]
        # Add ConvBlocks for each feature size
        for feature in features[1:]:
            # Use stride=1 for the last block, else stride=2
            layers.append(ConvBlock(in_c, feature, stride=1 if feature == features[-1] else 2))
            in_c = feature

        # Final convolution to produce a single-channel output
        layers.append(
            nn.Conv2d(
                in_c, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )

        # Combine all layers into a sequential module
        self.all = nn.Sequential(*layers)
    
    def forward(self, x):
        # Pass input through initial layer
        X = self.initial(x)
        # Pass through the rest of the network
        return self.all(X)