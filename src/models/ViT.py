from transformers import SegformerModel
import torch.nn as nn
from torchvision import transforms


def prepare_for_vit(x):
    transform = transforms.Compose([
        transforms.Resize((1024, 1280)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(x)


class MiTB0Encoder(nn.Module): 
    #backward pass is tested in the pytest unit test but not defined here.
    # it is a child class of the overall MiTB0UNet, which is tested in the same unit test
    def __init__(self, pretrained=True, freeze_backbone=False, base_ch=64):
        super().__init__()
        b = base_ch              # BUG 1 FIX: b was never assigned, proj layers all crashed
        self.base_ch = base_ch

        if pretrained:
            self.backbone = SegformerModel.from_pretrained("nvidia/mit-b0")
        else:
            from transformers import SegformerConfig
            self.backbone = SegformerModel(SegformerConfig())

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.proj1 = nn.Sequential(nn.Conv2d(32,  b,   kernel_size=1), nn.BatchNorm2d(b),   nn.ReLU(inplace=True))
        self.proj2 = nn.Sequential(nn.Conv2d(64,  2*b, kernel_size=1), nn.BatchNorm2d(2*b), nn.ReLU(inplace=True))
        self.proj3 = nn.Sequential(nn.Conv2d(160, 4*b, kernel_size=1), nn.BatchNorm2d(4*b), nn.ReLU(inplace=True))
        self.proj4 = nn.Sequential(nn.Conv2d(256, 8*b, kernel_size=1), nn.BatchNorm2d(8*b), nn.ReLU(inplace=True))

    def forward(self, x):          # BUG 2 FIX: forward/freeze/unfreeze were indented
        out = self.backbone(       # inside __init__ — they were never real methods
            pixel_values=x,
            output_hidden_states=True
        )

        f1, f2, f3, f4 = out.hidden_states

        p1 = self.proj1(f1)  # (B, b,   128, 128)
        p2 = self.proj2(f2)  # (B, 2b,   64,  64)
        p3 = self.proj3(f3)  # (B, 4b,   32,  32)
        p4 = self.proj4(f4)  # (B, 8b,   16,  16)

        return p1, p2, p3, p4     # BUG 3 FIX: was returning raw f1-f4, skipping
                                   # the projections entirely

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True