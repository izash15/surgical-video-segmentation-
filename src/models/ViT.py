#the integration and intiial starting of running the MiT B0 model for our ViT implementation

from transformers import SegformerModel
import torch.nn as nn
from torchvision import transforms

# The pre processing specific to the ViT
def prepare_for_vit(x):
    # x shape: (B, 3, H, W) — after this function, it should be (B, 3, 224, 224)
    # You can use torchvision.transforms for resizing and normalization
    transform = transforms.Compose([
        transforms.Resize((1024, 1280)), #this needs to be divisible by 32
        # ^^^ 512X512 will cause distortion current sizes are 1024X1280 (in order in accordance to the code) mess with larger sizes.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(x)

class MiTB0Encoder(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False, base_ch=64):
        super().__init__()
        self.base_channels = base_ch

        if pretrained:
            self.backbone = SegformerModel.from_pretrained("nvidia/mit-b0") # we are working on the pre-trained weights NOT training from scratch
        else:
            # Useful for offline/CI testing
            from transformers import SegformerConfig
            self.backbone = SegformerModel(SegformerConfig())
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.proj1 = nn.Sequential(nn.Conv2d(32,  b,   kernel_size=1), nn.BatchNorm2d(b),   nn.ReLU(inplace=True))
        self.proj2 = nn.Sequential(nn.Conv2d(64,  2*b, kernel_size=1), nn.BatchNorm2d(2*b), nn.ReLU(inplace=True))
        self.proj3 = nn.Sequential(nn.Conv2d(160, 4*b, kernel_size=1), nn.BatchNorm2d(4*b), nn.ReLU(inplace=True))
        self.proj4 = nn.Sequential(nn.Conv2d(256, 8*b, kernel_size=1), nn.BatchNorm2d(8*b), nn.ReLU(inplace=True))
    def forward(self, x):
        # x shape: (B, 3, H, W) — after prepare_for_vit()
        out = self.backbone(
            pixel_values=x,
            output_hidden_states=True
        )
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        # Returns tuple of 4 — one feature map per stage
        # You'll feed ALL 4 into the future bottleneck
        f1, f2, f3, f4 = out.hidden_states
        
        return f1, f2, f3, f4