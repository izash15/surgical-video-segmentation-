# check_mit_shapes.py  — updated
import torch
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.ViT import MiTB0Encoder

enc = MiTB0Encoder(pretrained=False, freeze_backbone=True, base_ch=64)
enc.eval()

x = torch.randn(2, 3, 512, 512)

with torch.no_grad():
    p1, p2, p3, p4 = enc(x)

print(p1.shape)  # expect (2,  64, 128, 128)
print(p2.shape)  # expect (2, 128,  64,  64)
print(p3.shape)  # expect (2, 256,  32,  32)
print(p4.shape)  # expect (2, 512,  16,  16)

# Confirm backbone params are actually frozen
frozen = all(not p.requires_grad for p in enc.backbone.parameters())
trainable = all(p.requires_grad for p in enc.proj1.parameters())
print(f"backbone frozen: {frozen}")      # expect True
print(f"proj1 trainable: {trainable}")   # expect True