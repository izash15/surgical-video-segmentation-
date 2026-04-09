import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from mmcv.ops import ModulatedDeformConv2d
from ViT import MiTB0Encoder, prepare_for_vit
from tripath import DoubleConv, Down, Up, OutConv, DCNBlock, DSConvBlockMMCV


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
    """

    def __init__(self, in_ch=3, num_classes=2, base_ch=64):
        super().__init__()
        b = base_ch

        # -------- 4 separate stems (full-res H/1) --------
        self.stem_cnn = DoubleConv(in_ch, b)
        self.stem_dcn = DoubleConv(in_ch, b)
        self.stem_dsc = DoubleConv(in_ch, b)  # keep DoubleConv at H/1 for cost/stability
        self.vit = MiTB0Encoder(pretrained=True, base_ch=base_ch) # one call of vit returns all 3 (h/2, h/4, h/8+deep)

        # -------- Encoders (four branches) --------
        # Stage H/2:  b -> 2b
        self.down1_cnn = Down(b, 2*b, block_cls=DoubleConv)
        self.down1_dcn = Down(b, 2*b, block_cls=DCNBlock)
        self.down1_dsc = Down(b, 2*b, block_cls=DSConvBlockMMCV)
        
        # Stage H/4:  2b -> 4b
        self.down2_cnn = Down(2*b, 4*b, block_cls=DoubleConv)
        self.down2_dcn = Down(2*b, 4*b, block_cls=DCNBlock)
        self.down2_dsc = Down(2*b, 4*b, block_cls=DSConvBlockMMCV)
        

        # Stage H/8:  4b -> 8b
        self.down3_cnn = Down(4*b, 8*b, block_cls=DoubleConv)
        self.down3_dcn = Down(4*b, 8*b, block_cls=DCNBlock)
        self.down3_dsc = Down(4*b, 8*b, block_cls=DSConvBlockMMCV)
       
        # Stage H/16 (bottleneck): 8b -> 16b
        self.down4_cnn = Down(8*b, 16*b, block_cls=DoubleConv)
        self.down4_dcn = Down(8*b, 16*b, block_cls=DCNBlock)
        self.down4_dsc = Down(8*b, 16*b, block_cls=DSConvBlockMMCV)

        # Stacked channels:
        #   skip1 (H/1):  3*b              (CNN+DCN+DSC only, ViT has no full-res stem)
        #   skip2 (H/2):  3*(2b)  = 6*b   (CNN+DCN+DSC only, ViT starts at H/4)
        #   skip4 (H/4):  4*(4b)  = 16*b  (CNN+DCN+DSC+ViT)
        #   skip8 (H/8):  4*(8b)  = 32*b  (CNN+DCN+DSC+ViT)
        #   bottleneck:   4*(16b) = 64*b  (CNN+DCN+DSC+ViT)

        self.up1 = Up(in_ch=48*b, skip_ch=24*b, out_ch=8*b)
        self.up2 = Up(in_ch=8*b,  skip_ch=12*b, out_ch=4*b)
        self.up3 = Up(in_ch=4*b,  skip_ch=6*b,  out_ch=b)
        self.up4 = Up(in_ch=b,    skip_ch=3*b,  out_ch=b)

        self.outc = OutConv(b, num_classes)

    def forward(self, x):
        # ----- Stems (H/1) -----
        s_c = self.stem_cnn(x)
        s_d = self.stem_dcn(x)
        s_s = self.stem_dsc(x)
        skip1 = torch.cat([s_c, s_d, s_s], dim=1)  # [B, 3b, H, W] — no ViT here


        # ----- Branch 1: CNN -----
        x2_c = self.down1_cnn(s_c)   # [B, 2b, H/2, W/2]
        x3_c = self.down2_cnn(x2_c)  # [B, 4b, H/4, W/4]
        x4_c = self.down3_cnn(x3_c)  # [B, 8b, H/8, W/8]
        x5_c = self.down4_cnn(x4_c)  # [B,16b, H/16,W/16]

        # ----- Branch 2: DCN -----
        x2_d = self.down1_dcn(s_d)
        x3_d = self.down2_dcn(x2_d)
        x4_d = self.down3_dcn(x3_d)
        x5_d = self.down4_dcn(x4_d)

        # ----- Branch 3: DSC (new) -----
        x2_s = self.down1_dsc(s_s)
        x3_s = self.down2_dsc(x2_s)
        x4_s = self.down3_dsc(x3_s)
        x5_s = self.down4_dsc(x4_s)

        # ----- Stack (concat) bottleneck + skips -----
        bottleneck = torch.cat([x5_c, x5_d, x5_s, v16], dim=1)  # [B, 64b, H/16, W/16]
        skip8      = torch.cat([x4_c, x4_d, x4_s, v8],  dim=1)  # [B, 32b, H/8,  W/8 ]
        skip4      = torch.cat([x3_c, x3_d, x3_s, v4],  dim=1)  # [B, 16b, H/4,  W/4 ]
        skip2      = torch.cat([x2_c, x2_d, x2_s],       dim=1)  # [B,  6b, H/2,  W/2 ] 

        # ----- Decoder -----
        x = self.up1(bottleneck, skip8)  # -> [B, 8b, H/8,  W/8 ]
        x = self.up2(x,          skip4)  # -> [B, 4b, H/4,  W/4 ]
        x = self.up3(x,          skip2)  # -> [B,  b, H/2,  W/2 ]
        x = self.up4(x,          skip1)  # -> [B,  b, H,    W   ]
        return self.outc(x)      # logits: [B, num_classes, H, W]

