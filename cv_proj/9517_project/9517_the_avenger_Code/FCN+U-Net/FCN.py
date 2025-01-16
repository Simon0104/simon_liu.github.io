import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, num_classes=4):
        super(FCN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))  # [batch_size, 64, 480, 480]
        x2 = F.relu(self.conv2(self.pool(x1)))  # [batch_size, 128, 240, 240]
        x3 = F.relu(self.conv3(self.pool(x2)))  # [batch_size, 256, 120, 120]
        x4 = F.relu(self.conv4(self.pool(x3)))  # [batch_size, 512, 60, 60]
        x5 = F.relu(self.conv5(self.pool(x4)))  # [batch_size, 1024, 30, 30]

        x4_up = F.relu(self.upconv4(x5))  # [batch_size, 512, 60, 60]
        x3_up = F.relu(self.upconv3(x4_up + x4))  # [batch_size, 256, 120, 120]
        x2_up = F.relu(self.upconv2(x3_up + x3))  # [batch_size, 128, 240, 240]
        x1_up = F.relu(self.upconv1(x2_up + x2))  # [batch_size, 64, 480, 480]

        output = self.final_conv(x1_up)  # [batch_size, num_classes, 480, 480]
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_output = nn.Conv2d(256, out_channels, kernel_size=1, stride=1)
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
        #print("out shape:", out.shape)
        out = self.conv1x1_output(out)
        return self.dropout(out)

class FCNWithASPP(nn.Module):
    def __init__(self, num_classes=4):
        super(FCNWithASPP, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.aspp = ASPP(128, 32)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  #  240x240

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  #    120x120

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  #  60x60

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  #  30x30

        x = self.aspp(x)

        x = F.relu(self.conv5(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  #  60x60

        x = F.relu(self.conv6(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  #  120x120

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  #  240x240
        x = self.final_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  #  480x480

        return x
