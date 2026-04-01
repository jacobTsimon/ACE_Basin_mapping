import torch
import torch.nn as nn


#complete model architecture overhaul
class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder (contracting path)
        self.conv1 = self.contract_block(in_channels, 32)
        self.conv2 = self.contract_block(32, 64)
        self.conv3 = self.contract_block(64, 128)
        self.conv4 = self.contract_block(128, 256)
        self.conv5 = self.contract_block(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # Decoder (expanding path)
        self.upconv6 = self.expand_block(1024, 512)
        self.upconv5 = self.expand_block(512 * 2, 256)
        self.upconv4 = self.expand_block(256 * 2, 128)
        self.upconv3 = self.expand_block(128 * 2, 64)
        self.upconv2 = self.expand_block(64 * 2, 32)
        self.upconv1 = self.expand_block(32 * 2, 32)

        # Final classification layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Bottleneck
        bottleneck = self.bottleneck(conv5)

        # Decoder with skip connections
        upconv6 = self.upconv6(bottleneck)

        # Use interpolation to ensure exact size match
        upconv6 = self._match_size(upconv6, conv5)
        upconv5 = self.upconv5(torch.cat([upconv6, conv5], 1))

        upconv5 = self._match_size(upconv5, conv4)
        upconv4 = self.upconv4(torch.cat([upconv5, conv4], 1))

        upconv4 = self._match_size(upconv4, conv3)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))

        upconv3 = self._match_size(upconv3, conv2)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))

        upconv2 = self._match_size(upconv2, conv1)
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        # Final output
        output = self.final_conv(upconv1)

        return output

    def _match_size(self, x, target):
        """Ensure x matches the spatial dimensions of target using interpolation."""
        if x.shape[2:] != target.shape[2:]:
            x = torch.nn.functional.interpolate(
                x,
                size=target.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        return x

    def contract_block(self, in_channels, out_channels):
        """
        Contracting block: 2 convolutions + batch norm + ReLU + dropout, then downsample with MaxPool
        """
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling by 2x
        )
        return contract

    def expand_block(self, in_channels, out_channels):
        """
        Expanding block: upsample, then 2 convolutions + batch norm + ReLU + dropout
        """
        expand = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),  # Upsampling by 2x
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return expand

