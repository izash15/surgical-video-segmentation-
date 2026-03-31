import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
from src.models.ViT import MiTB0Encoder


def test_encoder_initializes():
    model = MiTB0Encoder(pretrained=False, freeze_backbone=True, base_ch=64)
    assert model is not None
    assert hasattr(model, "backbone")
    assert hasattr(model, "proj1")
    assert hasattr(model, "proj2")
    assert hasattr(model, "proj3")
    assert hasattr(model, "proj4")


def test_output_shapes():
    model = MiTB0Encoder(pretrained=False, freeze_backbone=True, base_ch=64)
    model.eval()

    x = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        p1, p2, p3, p4 = model(x)

    assert p1.shape == (2, 64, 128, 128)
    assert p2.shape == (2, 128, 64, 64)
    assert p3.shape == (2, 256, 32, 32)
    assert p4.shape == (2, 512, 16, 16)


def test_backbone_is_frozen():
    model = MiTB0Encoder(pretrained=False, freeze_backbone=True, base_ch=64)

    backbone_frozen = all(not p.requires_grad for p in model.backbone.parameters())
    proj_trainable = all(p.requires_grad for p in model.proj1.parameters())

    assert backbone_frozen is True
    assert proj_trainable is True


def test_unfreeze_backbone():
    model = MiTB0Encoder(pretrained=False, freeze_backbone=True, base_ch=64)
    model.unfreeze_backbone()

    backbone_trainable = all(p.requires_grad for p in model.backbone.parameters())
    assert backbone_trainable is True


def test_forward_has_no_nans():
    model = MiTB0Encoder(pretrained=False, freeze_backbone=True, base_ch=64)
    model.eval()

    x = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        outputs = model(x)

    for out in outputs:
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


def test_backward_pass():
    model = MiTB0Encoder(pretrained=False, freeze_backbone=False, base_ch=64)
    model.train()

    x = torch.randn(2, 3, 512, 512)
    p1, p2, p3, p4 = model(x)

    loss = p1.mean() + p2.mean() + p3.mean() + p4.mean()
    loss.backward()

    grad_found = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_found = True
            break

    assert grad_found is True