from transformers import SegformerModel, SegformerForSemanticSegmentation
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def prepare_for_vit(x):
    transform = transforms.Compose([
        transforms.Resize((1024, 1280)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(x)


class MiTB0Encoder(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False, base_ch=64):
        super().__init__()
        b = base_ch

        if pretrained:
            self.backbone = SegformerModel.from_pretrained("nvidia/mit-b0")
        else:
            from transformers import SegformerConfig
            self.backbone = SegformerModel(SegformerConfig())

        if freeze_backbone:
            self.freeze_backbone()

        # Match existing UNet deeper scales
        self.proj_skip4 = nn.Sequential(
            nn.Conv2d(32, 4*b, kernel_size=1),
            nn.BatchNorm2d(4*b),
            nn.ReLU(inplace=True)
        )
        self.proj_skip8 = nn.Sequential(
            nn.Conv2d(64, 8*b, kernel_size=1),
            nn.BatchNorm2d(8*b),
            nn.ReLU(inplace=True)
        )
        self.proj_bottleneck = nn.Sequential(
            nn.Conv2d(160, 16*b, kernel_size=1),
            nn.BatchNorm2d(16*b),
            nn.ReLU(inplace=True)
        )

        # optional: deepest stage refinement
        self.proj_deep = nn.Sequential(
            nn.Conv2d(256, 16*b, kernel_size=1),
            nn.BatchNorm2d(16*b),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
        f1, f2, f3, f4 = out.hidden_states   # H/4, H/8, H/16, H/32

        v4 = self.proj_skip4(f1)             # [B, 4b, H/4, W/4]
        v8 = self.proj_skip8(f2)             # [B, 8b, H/8, W/8]
        v16 = self.proj_bottleneck(f3)       # [B,16b, H/16, W/16]

        deep = self.proj_deep(f4)            # [B,16b, H/32, W/32]
        deep = F.interpolate(deep, size=v16.shape[-2:], mode="bilinear", align_corners=False)

        v16 = v16 + deep                     # richer bottleneck

        return v4, v8, v16

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True