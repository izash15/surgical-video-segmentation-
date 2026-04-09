import torch
import torch.nn as nn

from models.tripath import DoubleConv, Down, Up, OutConv, DCNBlock, DSConvBlockMMCV
from models.ViT import MiTB0Encoder


class QuadPathUNetStacked(nn.Module):
    """
    Four-encoder UNet with stacked (concatenated) features:
      - Four separate stems (one per path) to provide a 4-way full-res skip.
      - Encoders:
          CNN path   : DoubleConv blocks
          DCN path   : DCNBlock blocks (your deformable impl)
          DSC path   : DSConvBlockMMCV blocks (new dynamic snake conv)
          ViT path   : MiTB0Encoder (pretrained Segformer backbone)
      - No fusion/gating: just torch.cat along C at each scale.
      - Decoder expects stacked bottleneck and stacked skips.
    Output: logits with num_classes channels (use sigmoid for multilabel).

    CNN/DCN/DSC branches provide:
      - skip1: H/1
      - skip2: H/2
      - skip4: H/4
      - skip8: H/8
      - bottleneck: H/16

    ViT branch provides:
      - v4:  H/4
      - v8:  H/8
      - v16: H/16

    So ViT is concatenated only into skip4, skip8, and bottleneck.
    """

    def __init__(self, in_ch=3, num_classes=2, base_ch=64, vit_pretrained=True, vit_freeze=False):
        super().__init__()
        b = base_ch

        # -------- CNN-style stems --------
        self.stem_cnn = DoubleConv(in_ch, b)
        self.stem_dcn = DoubleConv(in_ch, b)
        self.stem_dsc = DoubleConv(in_ch, b)

        # -------- CNN/DCN/DSC encoders --------
        self.down1_cnn = Down(b, 2 * b, block_cls=DoubleConv)
        self.down1_dcn = Down(b, 2 * b, block_cls=DCNBlock)
        self.down1_dsc = Down(b, 2 * b, block_cls=DSConvBlockMMCV)

        self.down2_cnn = Down(2 * b, 4 * b, block_cls=DoubleConv)
        self.down2_dcn = Down(2 * b, 4 * b, block_cls=DCNBlock)
        self.down2_dsc = Down(2 * b, 4 * b, block_cls=DSConvBlockMMCV)

        self.down3_cnn = Down(4 * b, 8 * b, block_cls=DoubleConv)
        self.down3_dcn = Down(4 * b, 8 * b, block_cls=DCNBlock)
        self.down3_dsc = Down(4 * b, 8 * b, block_cls=DSConvBlockMMCV)

        self.down4_cnn = Down(8 * b, 16 * b, block_cls=DoubleConv)
        self.down4_dcn = Down(8 * b, 16 * b, block_cls=DCNBlock)
        self.down4_dsc = Down(8 * b, 16 * b, block_cls=DSConvBlockMMCV)

        # -------- ViT branch --------
        self.vit = MiTB0Encoder(
            pretrained=vit_pretrained,
            freeze_backbone=vit_freeze,
            base_ch=b
        )

        # -------- Decoder --------
        # skip1: 3*b
        # skip2: 3*(2b) = 6*b
        # skip4: 3*(4b) + 4b = 16*b
        # skip8: 3*(8b) + 8b = 32*b
        # bottleneck: 3*(16b) + 16b = 64*b

        self.up1 = Up(in_ch=64 * b, skip_ch=32 * b, out_ch=8 * b)  # H/16 -> H/8
        self.up2 = Up(in_ch=8 * b,  skip_ch=16 * b, out_ch=4 * b)  # H/8  -> H/4
        self.up3 = Up(in_ch=4 * b,  skip_ch=6 * b,  out_ch=b)      # H/4  -> H/2
        self.up4 = Up(in_ch=b,      skip_ch=3 * b,  out_ch=b)      # H/2  -> H/1

        self.outc = OutConv(b, num_classes)

    def forward(self, x):
        # ----- CNN-style stems -----
        s_c = self.stem_cnn(x)   # [B, b, H, W]
        s_d = self.stem_dcn(x)   # [B, b, H, W]
        s_s = self.stem_dsc(x)   # [B, b, H, W]

        skip1 = torch.cat([s_c, s_d, s_s], dim=1)  # [B, 3b, H, W]

        # ----- CNN branch -----
        x2_c = self.down1_cnn(s_c)   # [B, 2b, H/2, W/2]
        x3_c = self.down2_cnn(x2_c)  # [B, 4b, H/4, W/4]
        x4_c = self.down3_cnn(x3_c)  # [B, 8b, H/8, W/8]
        x5_c = self.down4_cnn(x4_c)  # [B,16b, H/16,W/16]

        # ----- DCN branch -----
        x2_d = self.down1_dcn(s_d)
        x3_d = self.down2_dcn(x2_d)
        x4_d = self.down3_dcn(x3_d)
        x5_d = self.down4_dcn(x4_d)

        # ----- DSC branch -----
        x2_s = self.down1_dsc(s_s)
        x3_s = self.down2_dsc(x2_s)
        x4_s = self.down3_dsc(x3_s)
        x5_s = self.down4_dsc(x4_s)

        # ----- ViT branch -----
        v4, v8, v16 = self.vit(x)
        # expected:
        # v4  = [B,  4b, H/4,  W/4]
        # v8  = [B,  8b, H/8,  W/8]
        # v16 = [B, 16b, H/16, W/16]

        # ----- Concatenate -----
        skip2 = torch.cat([x2_c, x2_d, x2_s], dim=1)         # [B,  6b, H/2,  W/2]
        skip4 = torch.cat([x3_c, x3_d, x3_s, v4], dim=1)     # [B, 16b, H/4,  W/4]
        skip8 = torch.cat([x4_c, x4_d, x4_s, v8], dim=1)     # [B, 32b, H/8,  W/8]
        bottleneck = torch.cat([x5_c, x5_d, x5_s, v16], dim=1)  # [B, 64b, H/16, W/16]

        # ----- Decoder -----
        x = self.up1(bottleneck, skip8)   # -> [B, 8b, H/8, W/8]
        x = self.up2(x, skip4)            # -> [B, 4b, H/4, W/4]
        x = self.up3(x, skip2)            # -> [B, b,  H/2, W/2]
        x = self.up4(x, skip1)            # -> [B, b,  H,   W]
        return self.outc(x)