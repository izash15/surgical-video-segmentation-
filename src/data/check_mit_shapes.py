# check_mit_shapes.py
import torch
from transformers import SegformerModel, SegformerConfig

# Inline the encoder just enough to test — no need to import your full module
class MiTB0Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SegformerModel(SegformerConfig())  # random weights, no download

enc = MiTB0Encoder()
enc.eval()

x = torch.randn(2, 3, 512, 512)

with torch.no_grad():
    out = enc.backbone(pixel_values=x, output_hidden_states=True)

print(f"Number of hidden states: {len(out.hidden_states)}")
for i, h in enumerate(out.hidden_states):
    print(f"  stage {i+1}: {h.shape}")