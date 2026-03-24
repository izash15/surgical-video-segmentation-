#the integration and intiial starting of running the MiT B0 model for our ViT implementation

from transformers import SegformerModel
import torch.nn as nn
from torchvision import transforms

# The pre processing specific to the ViT
def prepare_for_vit(x):
    # x shape: (B, 3, H, W) — after this function, it should be (B, 3, 224, 224)
    # You can use torchvision.transforms for resizing and normalization
    transform = transforms.Compose([
        transforms.Resize((512, 512)) #this needs to be divisible by 32
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(x)

class MiTB0Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        if pretrained:
            self.backbone = SegformerModel.from_pretrained("nvidia/mit-b0")
        else:
            # Useful for offline/CI testing
            from transformers import SegformerConfig
            self.backbone = SegformerModel(SegformerConfig())
    
    def forward(self, x):
        # x shape: (B, 3, H, W) — after prepare_for_vit()
        out = self.backbone(
            pixel_values=x,
            output_hidden_states=True
        )
        
        # Returns tuple of 4 — one feature map per stage
        # You'll feed ALL 4 into the future bottleneck
        f1, f2, f3, f4 = out.hidden_states
        
        return f1, f2, f3, f4