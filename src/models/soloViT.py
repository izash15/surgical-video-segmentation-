from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from models.ViT import MiTB0Encoder, prepare_for_vit
 
 
class ViTSolo(MiTB0Encoder):
    """
    Standalone ViT benchmark model. Subclasses MiTB0Encoder so the backbone
    is identical to the stacked path, but swaps in Segformer's native All-MLP
    decode head instead of the custom projection layers.
 
    MiTB0Encoder (stacked):  image -> backbone -> proj layers -> v4, v8, v16
    ViTSolo       (solo):     image -> backbone -> All-MLP head -> logits [B, 1, H, W]
 
    Usage:
        model = ViTSolo(pretrained=True)
        logits = model(x)                   # [B, 1, H, W]
        preds  = torch.sigmoid(logits) > 0.5
    """
 
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False, num_classes: int = 1):
        # Initialise the parent with pretrained=False so it doesn't download
        # SegformerModel — we're going to replace self.backbone immediately below
        super().__init__(pretrained=False, freeze_backbone=False)
 
        # Swap backbone for the full encode+decode version
        if pretrained:
            self.backbone = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0",
                num_labels=num_classes,
                ignore_mismatched_sizes=True  # re-initialises head for num_classes
            )
        else:
            config = SegformerConfig.from_pretrained("nvidia/mit-b0")
            config.num_labels = num_classes
            self.backbone = SegformerForSemanticSegmentation(config)
 
        # Projection layers inherited from MiTB0Encoder are not used in solo
        # mode — delete them to keep the model clean and avoid confusion
        del self.proj_skip4
        del self.proj_skip8
        del self.proj_bottleneck
        del self.proj_deep
 
        if freeze_backbone:
            self.freeze_backbone()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
 
        x = prepare_for_vit(x)   # same preprocessing as stacked path
 
        out = self.backbone(pixel_values=x, return_dict=True)
        logits = out.logits       # [B, num_classes, H/4, W/4]
 
        # Upsample to original input resolution
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
 
    def freeze_backbone(self):
        # SegformerForSemanticSegmentation nests the encoder under .segformer
        for param in self.backbone.segformer.parameters():
            param.requires_grad = False
 
    def unfreeze_backbone(self):
        for param in self.backbone.segformer.parameters():
            param.requires_grad = True
 
 
# ── Quick shape sanity check ───────────────────────────────────────────────────
 
if __name__ == "__main__":
    model = ViTSolo(pretrained=False)
    model.eval()
 
    dummy = torch.randn(2, 3, 1024, 1280)
    with torch.no_grad():
        out = model(dummy)
 
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # expect [2, 1, 1024, 1280]
    assert out.shape == (2, 1, 1024, 1280), "Shape mismatch!"
    print("Shape check passed.")