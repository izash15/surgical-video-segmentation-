import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    """
    Up: explicitly specify (in_ch of decoder input, skip_ch from encoder, out_ch)
    We upsample to `skip_ch` channels, concat with skip, then DoubleConv to out_ch.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if shapes are off by 1 due to odd dims
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, base_ch=64):
        super().__init__()
        self.inc   = DoubleConv(in_ch,       base_ch)        # -> b
        self.down1 = Down(base_ch,           base_ch*2)      # -> 2b
        self.down2 = Down(base_ch*2,         base_ch*4)      # -> 4b
        self.down3 = Down(base_ch*4,         base_ch*8)      # -> 8b
        self.down4 = Down(base_ch*8,         base_ch*16)     # -> 16b (bottleneck)

        # Up path: (decoder_in_ch, skip_ch, out_ch)
        self.up1  = Up(base_ch*16, base_ch*8,  base_ch*8)    # out: 8b
        self.up2  = Up(base_ch*8,  base_ch*4,  base_ch*4)    # out: 4b
        self.up3  = Up(base_ch*4,  base_ch*2,  base_ch)      # out: b
        self.up4  = Up(base_ch,    base_ch,    base_ch)      # out: b

        self.outc = OutConv(base_ch, num_classes)

    def forward(self, x):
        x1 = self.inc(x)              # b
        x2 = self.down1(x1)           # 2b
        x3 = self.down2(x2)           # 4b
        x4 = self.down3(x3)           # 8b
        x5 = self.down4(x4)           # 16b

        x  = self.up1(x5, x4)         # 8b
        x  = self.up2(x,  x3)         # 4b
        x  = self.up3(x,  x2)         # b
        x  = self.up4(x,  x1)         # b
        return self.outc(x)
