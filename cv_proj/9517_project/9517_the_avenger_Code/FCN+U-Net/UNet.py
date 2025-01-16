import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes=4):
        super(UNet, self).__init__()

        self.enc_conv1 = self.double_conv(3, 16)
        self.enc_conv2 = self.double_conv(16, 32)
        self.enc_conv3 = self.double_conv(32, 64)
        self.enc_conv4 = self.double_conv(64, 128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.double_conv(128, 256)

        self.upconv4 = self.upconv(256, 128)
        self.dec_conv4 = self.double_conv(256, 128)
        self.upconv3 = self.upconv(128, 64)
        self.dec_conv3 = self.double_conv(128, 64)
        self.upconv2 = self.upconv(64, 32)
        self.dec_conv2 = self.double_conv(64, 32)
        self.upconv1 = self.upconv(32, 16)
        self.dec_conv1 = self.double_conv(32, 16)

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc_conv1(x)  # [batch_size, 64, 480, 480]
        enc2 = self.enc_conv2(self.pool(enc1))  # [batch_size, 128, 240, 240]
        enc3 = self.enc_conv3(self.pool(enc2))  # [batch_size, 256, 120, 120]
        enc4 = self.enc_conv4(self.pool(enc3))  # [batch_size, 512, 60, 60]

        bottleneck = self.bottleneck(self.pool(enc4))  # [batch_size, 1024, 30, 30]

        dec4 = self.upconv4(bottleneck)  # [batch_size, 512, 60, 60]
        dec4 = torch.cat((dec4, enc4), dim=1)  #   [batch_size, 1024, 60, 60]
        dec4 = self.dec_conv4(dec4)  # [batch_size, 512, 60, 60]

        dec3 = self.upconv3(dec4)  # [batch_size, 256, 120, 120]
        dec3 = torch.cat((dec3, enc3), dim=1)  #  [batch_size, 512, 120, 120]
        dec3 = self.dec_conv3(dec3)  # [batch_size, 256, 120, 120]

        dec2 = self.upconv2(dec3)  # [batch_size, 128, 240, 240]
        dec2 = torch.cat((dec2, enc2), dim=1)  #  [batch_size, 256, 240, 240]
        dec2 = self.dec_conv2(dec2)  # [batch_size, 128, 240, 240]

        dec1 = self.upconv1(dec2)  # [batch_size, 64, 480, 480]
        dec1 = torch.cat((dec1, enc1), dim=1)  #  [batch_size, 128, 480, 480]
        dec1 = self.dec_conv1(dec1)  # [batch_size, 64, 480, 480]

        output = self.final_conv(dec1)  # [batch_size, num_classes, 480, 480]
        return output


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_output = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        size = x.shape[2:]

        out1 = self.atrous_block1(x)
        out2 = self.atrous_block6(x)
        out3 = self.atrous_block12(x)
        out4 = self.atrous_block18(x)
        out5 = self.global_avg_pool(x)
        out5 = F.interpolate(out5, size=size, mode='bilinear', align_corners=True)

        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.conv1x1_output(out)
        return self.dropout(out)

class UNetWithASPP(nn.Module):
    def __init__(self, num_classes=4):
        super(UNetWithASPP, self).__init__()

        self.enc1 = self.double_conv(3, 16)
        self.enc2 = self.double_conv(16, 32)
        self.enc3 = self.double_conv(32, 64)
        self.enc4 = self.double_conv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.aspp = ASPP(128, 128)

        self.up1 = self.upconv(128, 128)
        self.dec1 = self.double_conv(256, 128)
        self.up2 = self.upconv(128, 64)
        self.dec2 = self.double_conv(128, 64)
        self.up3 = self.upconv(64, 32)
        self.dec3 = self.double_conv(64, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)  # [batch_size, 64, 480, 480]
        enc2 = self.enc2(self.pool(enc1))  # [batch_size, 128, 240, 240]
        enc3 = self.enc3(self.pool(enc2))  # [batch_size, 256, 120, 120]
        enc4 = self.enc4(self.pool(enc3))  # [batch_size, 512, 60, 60]

        x = self.aspp(self.pool(enc4))  # [batch_size, 256, 30, 30]

        x = self.up1(x)  # [batch_size, 256, 60, 60]
        x = torch.cat([x, enc4], dim=1)  #
        x = self.dec1(x)  # [batch_size, 256, 60, 60]

        x = self.up2(x)  # [batch_size, 128, 120, 120]
        x = torch.cat([x, enc3], dim=1)
        x = self.dec2(x)  # [batch_size, 128, 120, 120]

        x = self.up3(x)  # [batch_size, 64, 240, 240]
        x = torch.cat([x, enc2], dim=1)
        x = self.dec3(x)  # [batch_size, 64, 240, 240]

        x = self.final_conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))  # [batch_size, num_classes, 480, 480]
        return x
