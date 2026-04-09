import torch
from ViTStacked import QuadPathUNetStacked

def test_smoke():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") 
    
    model = QuadPathUNetStacked(in_ch=3, num_classes=2, base_ch=64)
    model.eval()

    # Fake batch: 2 images, 3 channels, 1024x1280
    x = torch.randn(2, 3, 1024, 1280)

    with torch.no_grad():
        out = model(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 2, 1024, 1280), f"Unexpected output shape: {out.shape}"
    print("PASSED")

if __name__ == "__main__":
    test_smoke()