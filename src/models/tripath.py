import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from mmcv.ops import ModulatedDeformConv2d
# -----------------------------
# Basic building blocks
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class Down(nn.Module):
    """MaxPool(2) + Block(in_ch->out_ch), where Block can be DoubleConv / DCN / NSConv variants."""
    def __init__(self, in_ch, out_ch, block_cls=DoubleConv):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = block_cls(in_ch, out_ch)
    def forward(self, x): return self.block(self.pool(x))


class Up(nn.Module):
    """
    Up: (in_ch of decoder input, skip_ch after fusion, out_ch)
    Up-conv to skip_ch, concat with fused skip (skip_ch), then DoubleConv(2*skip_ch -> out_ch)
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if shapes are off by 1
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    def forward(self, x): return self.conv(x)

# -----------------------------
# Optional DCN / NSConv blocks
# (fallback to DoubleConv if not available)
# -----------------------------
# -----------------------------
# Snake activation (learnable frequency)
#   Snake(x) = x + (1/α) * sin^2(αx)
#   α is learned per-channel.
# -----------------------------
class Snake(nn.Module):
    def __init__(self, channels: int, init_alpha: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), init_alpha))
        self.eps = eps

    def forward(self, x):
        alpha = self.alpha.abs() + self.eps  # keep α > 0
        return x + (torch.sin(alpha * x) ** 2) / alpha


# -----------------------------
# NSConvBlock (Dynamic Snake Convolution)
# Two Conv-BN-Snake layers with learnable α per channel.
# -----------------------------
class NSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=bias)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act1  = Snake(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, bias=bias)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act2  = Snake(out_ch)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


# -----------------------------
# DCNBlock (true Deformable Conv) via mmcv
# Falls back to DoubleConv if mmcv/ops not available.
# -----------------------------
class DoubleConv(nn.Module):
    # keep your original DoubleConv for fallback / reuse
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.net(x)


class DCNBlock(nn.Module):
    """
    Two-layer DCN block:
      - Uses mmcv.ops.DeformConv2dPack (or ModulatedDeformConv2dPack if you prefer)
      - Each DCN is followed by BN + ReLU
    If mmcv isn't available, falls back to DoubleConv.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=False, use_modulated=False):
        super().__init__()
        self._using_dcn = False
        self.impl = None

        try:
            from mmcv.ops import DeformConv2dPack
            if use_modulated:
                # If you want modulated DCN instead, uncomment below (requires mmcv w/ modulated op)
                # from mmcv.ops import ModulatedDeformConv2dPack as DCNPack
                # DCNPackToUse = DCNPack
                DCNPackToUse = DeformConv2dPack  # placeholder, switch if using modulated
            else:
                DCNPackToUse = DeformConv2dPack

            self.conv1 = DCNPackToUse(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias)
            self.bn1   = nn.BatchNorm2d(out_ch)
            self.conv2 = DCNPackToUse(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias)
            self.bn2   = nn.BatchNorm2d(out_ch)
            self.relu  = nn.ReLU(inplace=True)
            self._using_dcn = True

        except Exception as e:
            warnings.warn(
                f"[DCNBlock] mmcv with DCN ops not available ({e}). "
                "Falling back to standard DoubleConv."
            )
            self.impl = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        if self._using_dcn:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            return x
        else:
            return self.impl(x)
# -----------------------------
# Gated fusion of 3 branches
# -----------------------------
class GateFuse3(nn.Module):
    """
    Align the three path features to out_ch, predict per-location soft weights, and fuse.
    Produces a fused skip with channels = out_ch.
    """
    def __init__(self, in_ch_each, out_ch):
        super().__init__()
        c = in_ch_each
        self.align_a = nn.Conv2d(c, out_ch, 1, bias=False)
        self.align_b = nn.Conv2d(c, out_ch, 1, bias=False)
        self.align_c = nn.Conv2d(c, out_ch, 1, bias=False)
        self.gate = nn.Conv2d(out_ch * 3, 3, 1)  # logits for weights
        self.post = nn.Conv2d(out_ch, out_ch, 1, bias=False)  # optional smoothing

    def forward(self, fa, fb, fc):
        fa = self.align_a(fa)
        fb = self.align_b(fb)
        fc = self.align_c(fc)
        g_logits = self.gate(torch.cat([fa, fb, fc], dim=1))
        g = torch.softmax(g_logits, dim=1)  # [B,3,H,W]
        fused = g[:, 0:1] * fa + g[:, 1:2] * fb + g[:, 2:3] * fc
        return self.post(fused)

# -----------------------------
# Tri-Path UNet
# -----------------------------
# --- Swap NSConv path → DSConv path; keep fusions unchanged ---
class TriPathUNet(nn.Module):
    """
    Three-encoder UNet:
      - Shared stem (DoubleConv in_ch -> b) !!!Fused
      - Encoder paths:
          CNN path   : DoubleConv blocks
          DCN path   : DCNBlock blocks (your deformable impl)
          DSConv path: DSConvBlockMMCV blocks (new)
      - Gated fusion at each skip scale + bottleneck
      - Standard UNet decoder
    Output: logits with num_classes channels (use sigmoid for multilabel)
    """
    def __init__(self, in_ch=3, num_classes=2, base_ch=64):
        super().__init__()
        b = base_ch

        # Shared stem (full-res skip)
        self.stem = DoubleConv(in_ch, b)  # -> x1_shared

        # -------- Encoders (three branches) --------
        # Stage H/2:  b -> 2b
        self.down1_cnn = Down(b,  2*b, block_cls=DoubleConv)
        self.down1_dcn = Down(b,  2*b, block_cls=DCNBlock)
        self.down1_dsc = Down(b,  2*b, block_cls=DSConvBlockMMCV)   # <-- was NSConvBlock

        # Stage H/4:  2b -> 4b
        self.down2_cnn = Down(2*b, 4*b, block_cls=DoubleConv)
        self.down2_dcn = Down(2*b, 4*b, block_cls=DCNBlock)
        self.down2_dsc = Down(2*b, 4*b, block_cls=DSConvBlockMMCV)  # <-- was NSConvBlock

        # Stage H/8:  4b -> 8b
        self.down3_cnn = Down(4*b, 8*b, block_cls=DoubleConv)
        self.down3_dcn = Down(4*b, 8*b, block_cls=DCNBlock)
        self.down3_dsc = Down(4*b, 8*b, block_cls=DSConvBlockMMCV)  # <-- was NSConvBlock

        # Stage H/16 (bottleneck): 8b -> 16b
        self.down4_cnn = Down(8*b,  16*b, block_cls=DoubleConv)
        self.down4_dcn = Down(8*b,  16*b, block_cls=DCNBlock)
        self.down4_dsc = Down(8*b,  16*b, block_cls=DSConvBlockMMCV) # <-- was NSConvBlock

        # -------- Fusions (unchanged) --------
        self.fuse_2b  = GateFuse3(in_ch_each=2*b,  out_ch=2*b)   # H/2
        self.fuse_4b  = GateFuse3(in_ch_each=4*b,  out_ch=4*b)   # H/4
        self.fuse_8b  = GateFuse3(in_ch_each=8*b,  out_ch=8*b)   # H/8
        self.fuse_16b = GateFuse3(in_ch_each=16*b, out_ch=16*b)  # bottleneck

        # -------- Decoder (unchanged) --------
        self.up1 = Up(16*b, 8*b,  8*b)  # from fused bottleneck, skip=fused 8b
        self.up2 = Up(8*b,  4*b,  4*b)
        self.up3 = Up(4*b,  2*b,  b)
        self.up4 = Up(b,    b,    b)    # last skip = shared stem
        self.outc = OutConv(b, num_classes)

    def forward(self, x):
        x1_shared = self.stem(x)  # [B,b,H,W]

        # Branch 1: CNN
        x2_c = self.down1_cnn(x1_shared)
        x3_c = self.down2_cnn(x2_c)
        x4_c = self.down3_cnn(x3_c)
        x5_c = self.down4_cnn(x4_c)

        # Branch 2: DCN
        x2_d = self.down1_dcn(x1_shared)
        x3_d = self.down2_dcn(x2_d)
        x4_d = self.down3_dcn(x3_d)
        x5_d = self.down4_dcn(x4_d)

        # Branch 3: DSConv (new)
        x2_s = self.down1_dsc(x1_shared)
        x3_s = self.down2_dsc(x2_s)
        x4_s = self.down3_dsc(x3_s)
        x5_s = self.down4_dsc(x4_s)

        # Fusions
        bottleneck = self.fuse_16b(x5_c, x5_d, x5_s)  # [B,16b,H/16,W/16]
        skip8      = self.fuse_8b (x4_c, x4_d, x4_s)  # [B, 8b,H/8 ,W/8 ]
        skip4      = self.fuse_4b (x3_c, x3_d, x3_s)  # [B, 4b,H/4 ,W/4 ]
        skip2      = self.fuse_2b (x2_c, x2_d, x2_s)  # [B, 2b,H/2 ,W/2 ]
        skip1      = x1_shared

        # Decoder
        x = self.up1(bottleneck, skip8)
        x = self.up2(x,          skip4)
        x = self.up3(x,          skip2)
        x = self.up4(x,          skip1)
        return self.outc(x)


class TriPathUNetStacked(nn.Module):
    """
    Three-encoder UNet with stacked (concatenated) features:
      - Three separate stems (one per path) to provide a 3-way full-res skip.
      - Encoders:
          CNN path   : DoubleConv blocks
          DCN path   : DCNBlock blocks (your deformable impl)
          DSC path   : DSConvBlockMMCV blocks (new dynamic snake conv)
      - No fusion/gating: just torch.cat along C at each scale.
      - Decoder expects stacked bottleneck and stacked skips.
    Output: logits with num_classes channels (use sigmoid for multilabel).
    """

    def __init__(self, in_ch=3, num_classes=2, base_ch=64):
        super().__init__()
        b = base_ch

        # -------- 3 separate stems (full-res H/1) --------
        self.stem_cnn = DoubleConv(in_ch, b)
        self.stem_dcn = DoubleConv(in_ch, b)
        self.stem_dsc = DoubleConv(in_ch, b)  # keep DoubleConv at H/1 for cost/stability

        # -------- Encoders (three branches) --------
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

        # -------- Decoder (channel sizes must match stacked skips) --------
        # Stacked channels:
        #   skip1 (H/1):  3*b
        #   skip2 (H/2):  3*(2b)  = 6*b
        #   skip4 (H/4):  3*(4b)  = 12*b
        #   skip8 (H/8):  3*(8b)  = 24*b
        #   bottleneck:   3*(16b) = 48*b
        self.up1 = Up(in_ch=48*b, skip_ch=24*b, out_ch=8*b)  # H/16 -> H/8
        self.up2 = Up(in_ch=8*b,  skip_ch=12*b, out_ch=4*b)  # H/8  -> H/4
        self.up3 = Up(in_ch=4*b,  skip_ch=6*b,  out_ch=b)    # H/4  -> H/2
        self.up4 = Up(in_ch=b,    skip_ch=3*b,  out_ch=b)    # H/2  -> H/1

        self.outc = OutConv(b, num_classes)

    def forward(self, x):
        # ----- Stems (H/1) -----
        s_c = self.stem_cnn(x)   # [B, b, H, W]
        s_d = self.stem_dcn(x)   # [B, b, H, W]
        s_s = self.stem_dsc(x)   # [B, b, H, W]
        skip1 = torch.cat([s_c, s_d, s_s], dim=1)  # [B, 3b, H, W]

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
        bottleneck = torch.cat([x5_c, x5_d, x5_s], dim=1)  # [B, 48b, H/16, W/16]
        skip8      = torch.cat([x4_c, x4_d, x4_s], dim=1)  # [B, 24b, H/8,  W/8 ]
        skip4      = torch.cat([x3_c, x3_d, x3_s], dim=1)  # [B, 12b, H/4,  W/4 ]
        skip2      = torch.cat([x2_c, x2_d, x2_s], dim=1)  # [B,  6b, H/2,  W/2 ]

        # ----- Decoder -----
        x = self.up1(bottleneck, skip8)  # -> [B, 8b, H/8,  W/8 ]
        x = self.up2(x,          skip4)  # -> [B, 4b, H/4,  W/4 ]
        x = self.up3(x,          skip2)  # -> [B,  b, H/2,  W/2 ]
        x = self.up4(x,          skip1)  # -> [B,  b, H,    W   ]
        return self.outc(x)      # logits: [B, num_classes, H, W]


# --- Patch your DSConv to be numerically safe ---
class DSConvMMCV(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1,
                 bias=False, use_mask=True, offset_scale=0.25, groups_gn=8):
        super().__init__()
        from mmcv.ops import ModulatedDeformConv2d

        self.k = 3
        self.use_mask = use_mask
        self.offset_scale = offset_scale

        head_out = 2 + (1 if use_mask else 0)  # Δx, Δy, (mask)
        self.offset_head = nn.Conv2d(in_ch, head_out, 3, padding=1)

        self.dcn = ModulatedDeformConv2d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=self.k, stride=stride, padding=padding, dilation=dilation,
            deform_groups=1, bias=bias
        )
        # Inits: near-identity conv; zero offsets & mask bias to ~0.5 after sigmoid
        nn.init.kaiming_normal_(self.dcn.weight, mode='fan_out', nonlinearity='relu')
        if self.dcn.bias is not None: nn.init.zeros_(self.dcn.bias)
        nn.init.zeros_(self.offset_head.weight)
        with torch.no_grad():
            self.offset_head.bias.zero_()
            if self.use_mask:
                self.offset_head.bias[2::3].fill_(0.0)  # sigmoid(0)=0.5

        # Use GroupNorm instead of BN to avoid running-stat issues
        gn_groups = min(groups_gn, out_ch)
        self.norm = nn.GroupNorm(gn_groups, out_ch)
        self.act  = nn.ReLU(inplace=True)

    @staticmethod
    def _tamed(dx, dy, scale):
        # Bound offsets: keeps DCN sampling stable
        dx = scale * torch.tanh(dx)
        dy = scale * torch.tanh(dy)
        return dx, dy

    def _serpentine_offsets(self, dx, dy):
        z = torch.zeros_like(dx)
        offs = torch.cat([
            z,  z,
            z,  +dy,
            z,  -dy,
            +dx, z,
            z,  z,
            -dx, z,
            +dx,+dy,
            +dx,-dy,
            -dx,+dy,
        ], dim=1)
        return offs

    def forward(self, x):
        n, c, h, w = x.shape
        head = self.offset_head(x)
        if self.use_mask:
            dx, dy, m = head[:, :1], head[:, 1:2], head[:, 2:3]
            mask = torch.sigmoid(m).expand(n, self.k*self.k, h, w)
        else:
            dx, dy = head[:, :1], head[:, 1:2]
            mask = torch.ones((n, self.k*self.k, h, w), device=x.device, dtype=x.dtype)

        dx, dy = self._tamed(dx, dy, self.offset_scale)
        offsets = self._serpentine_offsets(dx, dy)

        y = self.dcn(x, offsets, mask)
        return self.act(self.norm(y))

class DSConvBlockMMCV(nn.Module):
    """Two DSConvMMCV layers with BN+ReLU (already inside each layer)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.ds1 = DSConvMMCV_Safe(in_ch, out_ch)
        self.ds2 = DSConvMMCV_Safe(out_ch, out_ch)

    def forward(self, x):
        x = self.ds1(x)
        x = self.ds2(x)
        return x


if __name__ == "__main__":
    # Quick sanity check
    model = TriPathUNet(in_ch=3, num_classes=5, base_ch=64)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print("Output:", y.shape)  # [2, 5, 512, 512]


class DSConvMMCV_Safe(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1, dilation=1,
                 bias=False, use_mask=True, offset_scale=0.25, groups_gn=8):
        super().__init__()
        from mmcv.ops import ModulatedDeformConv2d
        self.k = 3
        self.use_mask = use_mask
        self.offset_scale = float(offset_scale)

        # IMPORTANT: make offset_head use the SAME stride/pad/dilation as DCN,
        # so offset/mask spatial size equals DCN output spatial size.
        head_out = 2 + (1 if use_mask else 0)
        self.offset_head = nn.Conv2d(in_ch, head_out, kernel_size=self.k,
                                     stride=stride, padding=padding, dilation=dilation)

        self.dcn = ModulatedDeformConv2d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=self.k, stride=stride, padding=padding, dilation=dilation,
            deform_groups=1, bias=bias
        )

        nn.init.kaiming_normal_(self.dcn.weight, mode='fan_out', nonlinearity='relu')
        if self.dcn.bias is not None:
            nn.init.zeros_(self.dcn.bias)
        nn.init.zeros_(self.offset_head.weight)
        with torch.no_grad():
            self.offset_head.bias.zero_()
            # if you want mask ~0.5 at init, ensure use_mask and set bias channel 2 to 0
            if self.use_mask and head_out >= 3:
                self.offset_head.bias[2].fill_(0.0)  # sigmoid(0)=0.5

        # GroupNorm (ensure divisibility)
        gn_groups = min(groups_gn, out_ch)
        if out_ch % gn_groups != 0:
            gn_groups = 1
        self.norm = nn.GroupNorm(gn_groups, out_ch)
        self.act  = nn.ReLU(inplace=True)

    @staticmethod
    def _tamed(dx, dy, scale: float):
        return scale * torch.tanh(dx), scale * torch.tanh(dy)

    def _serpentine_offsets(self, dx, dy):
        z = torch.zeros_like(dx)
        offs = torch.cat([
            z,  z,
            z,  +dy,
            z,  -dy,
            +dx, z,
            z,  z,
            -dx, z,
            +dx,+dy,
            +dx,-dy,
            -dx,+dy,
        ], dim=1)
        return offs

    def forward(self, x):
        # Make sure DCN sees contiguous float32 CUDA tensors
        x = x.contiguous().float()

        head = self.offset_head(x).contiguous().float()
        if self.use_mask:
            dx, dy, m = head[:, :1], head[:, 1:2], head[:, 2:3]
            mask = torch.sigmoid(m)
        else:
            dx, dy = head[:, :1], head[:, 1:2]
            # create a real tensor, not a broadcasted view
            mask = torch.ones((x.size(0), 1, head.size(2), head.size(3)),
                              device=x.device, dtype=x.dtype)

        dx, dy = self._tamed(dx, dy, self.offset_scale)
        offsets = self._serpentine_offsets(dx, dy).contiguous().float()   # (N, 18, H_out, W_out)

        # expand mask to (N,9,H_out,W_out) as a real contiguous tensor
        mask = mask.repeat(1, self.k*self.k, 1, 1).contiguous().float()   # (N, 9, H_out, W_out)

        # final safety checks (cheap)
        assert offsets.shape[1] == 2 * self.k * self.k
        assert mask.shape[1]    == self.k * self.k
        assert x.is_cuda and offsets.is_cuda and mask.is_cuda

        y = self.dcn(x, offsets, mask)          # DCNv2 op
        y = self.norm(y)
        y = self.act(y)
        return y

class TriPathCNN(nn.Module):
    """
    Three-encoder UNet:
      - Shared stem (DoubleConv in_ch -> b) !!!Fused
      - Encoder paths:
          CNN path   : DoubleConv blocks
          DCN path   : DCNBlock blocks (your deformable impl)
          DSConv path: DSConvBlockMMCV blocks (new)
      - Gated fusion at each skip scale + bottleneck
      - Standard UNet decoder
    Output: logits with num_classes channels (use sigmoid for multilabel)
    """
    def __init__(self, in_ch=3, num_classes=2, base_ch=64):
        super().__init__()
        b = base_ch

        # Shared stem (full-res skip)
        self.stem = DoubleConv(in_ch, b)  # -> x1_shared

        # -------- Encoders (three branches) --------
        # Stage H/2:  b -> 2b
        self.down1_cnn = Down(b,  2*b, block_cls=DoubleConv)
        self.down1_dcn = Down(b,  2*b, block_cls=DoubleConv)
        self.down1_dsc = Down(b,  2*b, block_cls=DoubleConv)   # <-- was NSConvBlock

        # Stage H/4:  2b -> 4b
        self.down2_cnn = Down(2*b, 4*b, block_cls=DoubleConv)
        self.down2_dcn = Down(2*b, 4*b, block_cls=DoubleConv)
        self.down2_dsc = Down(2*b, 4*b, block_cls=DoubleConv)  # <-- was NSConvBlock

        # Stage H/8:  4b -> 8b
        self.down3_cnn = Down(4*b, 8*b, block_cls=DoubleConv)
        self.down3_dcn = Down(4*b, 8*b, block_cls=DoubleConv)
        self.down3_dsc = Down(4*b, 8*b, block_cls=DoubleConv)  # <-- was NSConvBlock

        # Stage H/16 (bottleneck): 8b -> 16b
        self.down4_cnn = Down(8*b,  16*b, block_cls=DoubleConv)
        self.down4_dcn = Down(8*b,  16*b, block_cls=DoubleConv)
        self.down4_dsc = Down(8*b,  16*b, block_cls=DoubleConv) # <-- was NSConvBlock

        # -------- Fusions (unchanged) --------
        self.fuse_2b  = GateFuse3(in_ch_each=2*b,  out_ch=2*b)   # H/2
        self.fuse_4b  = GateFuse3(in_ch_each=4*b,  out_ch=4*b)   # H/4
        self.fuse_8b  = GateFuse3(in_ch_each=8*b,  out_ch=8*b)   # H/8
        self.fuse_16b = GateFuse3(in_ch_each=16*b, out_ch=16*b)  # bottleneck

        # -------- Decoder (unchanged) --------
        self.up1 = Up(16*b, 8*b,  8*b)  # from fused bottleneck, skip=fused 8b
        self.up2 = Up(8*b,  4*b,  4*b)
        self.up3 = Up(4*b,  2*b,  b)
        self.up4 = Up(b,    b,    b)    # last skip = shared stem
        self.outc = OutConv(b, num_classes)

    def forward(self, x):
        x1_shared = self.stem(x)  # [B,b,H,W]

        # Branch 1: CNN
        x2_c = self.down1_cnn(x1_shared)
        x3_c = self.down2_cnn(x2_c)
        x4_c = self.down3_cnn(x3_c)
        x5_c = self.down4_cnn(x4_c)

        # Branch 2: DCN
        x2_d = self.down1_dcn(x1_shared)
        x3_d = self.down2_dcn(x2_d)
        x4_d = self.down3_dcn(x3_d)
        x5_d = self.down4_dcn(x4_d)

        # Branch 3: DSConv (new)
        x2_s = self.down1_dsc(x1_shared)
        x3_s = self.down2_dsc(x2_s)
        x4_s = self.down3_dsc(x3_s)
        x5_s = self.down4_dsc(x4_s)

        # Fusions
        bottleneck = self.fuse_16b(x5_c, x5_d, x5_s)  # [B,16b,H/16,W/16]
        skip8      = self.fuse_8b (x4_c, x4_d, x4_s)  # [B, 8b,H/8 ,W/8 ]
        skip4      = self.fuse_4b (x3_c, x3_d, x3_s)  # [B, 4b,H/4 ,W/4 ]
        skip2      = self.fuse_2b (x2_c, x2_d, x2_s)  # [B, 2b,H/2 ,W/2 ]
        skip1      = x1_shared

        # Decoder
        x = self.up1(bottleneck, skip8)
        x = self.up2(x,          skip4)
        x = self.up3(x,          skip2)
        x = self.up4(x,          skip1)
        return self.outc(x)

class TriPathDSConv(nn.Module):
    """
    Three-encoder UNet:
      - Shared stem (DoubleConv in_ch -> b) !!!Fused
      - Encoder paths:
          CNN path   : DoubleConv blocks
          DCN path   : DCNBlock blocks (your deformable impl)
          DSConv path: DSConvBlockMMCV blocks (new)
          DSConvBlockMMCV

      - Gated fusion at each skip scale + bottleneck
      - Standard UNet decoder
    Output: logits with num_classes channels (use sigmoid for multilabel)
    """
    def __init__(self, in_ch=3, num_classes=2, base_ch=64):
        super().__init__()
        b = base_ch

        # Shared stem (full-res skip)
        self.stem = DoubleConv(in_ch, b)  # -> x1_shared

        # -------- Encoders (three branches) --------
        # Stage H/2:  b -> 2b
        self.down1_cnn = Down(b,  2*b, block_cls=DSConvBlockMMCV)
        self.down1_dcn = Down(b,  2*b, block_cls=DSConvBlockMMCV)
        self.down1_dsc = Down(b,  2*b, block_cls=DSConvBlockMMCV)   # <-- was NSConvBlock

        # Stage H/4:  2b -> 4b
        self.down2_cnn = Down(2*b, 4*b, block_cls=DSConvBlockMMCV)
        self.down2_dcn = Down(2*b, 4*b, block_cls=DSConvBlockMMCV)
        self.down2_dsc = Down(2*b, 4*b, block_cls=DSConvBlockMMCV)  # <-- was NSConvBlock

        # Stage H/8:  4b -> 8b
        self.down3_cnn = Down(4*b, 8*b, block_cls=DSConvBlockMMCV)
        self.down3_dcn = Down(4*b, 8*b, block_cls=DSConvBlockMMCV)
        self.down3_dsc = Down(4*b, 8*b, block_cls=DSConvBlockMMCV)  # <-- was NSConvBlock

        # Stage H/16 (bottleneck): 8b -> 16b
        self.down4_cnn = Down(8*b,  16*b, block_cls=DSConvBlockMMCV)
        self.down4_dcn = Down(8*b,  16*b, block_cls=DSConvBlockMMCV)
        self.down4_dsc = Down(8*b,  16*b, block_cls=DSConvBlockMMCV) # <-- was NSConvBlock

        # -------- Fusions (unchanged) --------
        self.fuse_2b  = GateFuse3(in_ch_each=2*b,  out_ch=2*b)   # H/2
        self.fuse_4b  = GateFuse3(in_ch_each=4*b,  out_ch=4*b)   # H/4
        self.fuse_8b  = GateFuse3(in_ch_each=8*b,  out_ch=8*b)   # H/8
        self.fuse_16b = GateFuse3(in_ch_each=16*b, out_ch=16*b)  # bottleneck

        # -------- Decoder (unchanged) --------
        self.up1 = Up(16*b, 8*b,  8*b)  # from fused bottleneck, skip=fused 8b
        self.up2 = Up(8*b,  4*b,  4*b)
        self.up3 = Up(4*b,  2*b,  b)
        self.up4 = Up(b,    b,    b)    # last skip = shared stem
        self.outc = OutConv(b, num_classes)

    def forward(self, x):
        x1_shared = self.stem(x)  # [B,b,H,W]

        # Branch 1: CNN
        x2_c = self.down1_cnn(x1_shared)
        x3_c = self.down2_cnn(x2_c)
        x4_c = self.down3_cnn(x3_c)
        x5_c = self.down4_cnn(x4_c)

        # Branch 2: DCN
        x2_d = self.down1_dcn(x1_shared)
        x3_d = self.down2_dcn(x2_d)
        x4_d = self.down3_dcn(x3_d)
        x5_d = self.down4_dcn(x4_d)

        # Branch 3: DSConv (new)
        x2_s = self.down1_dsc(x1_shared)
        x3_s = self.down2_dsc(x2_s)
        x4_s = self.down3_dsc(x3_s)
        x5_s = self.down4_dsc(x4_s)

        # Fusions
        bottleneck = self.fuse_16b(x5_c, x5_d, x5_s)  # [B,16b,H/16,W/16]
        skip8      = self.fuse_8b (x4_c, x4_d, x4_s)  # [B, 8b,H/8 ,W/8 ]
        skip4      = self.fuse_4b (x3_c, x3_d, x3_s)  # [B, 4b,H/4 ,W/4 ]
        skip2      = self.fuse_2b (x2_c, x2_d, x2_s)  # [B, 2b,H/2 ,W/2 ]
        skip1      = x1_shared

        # Decoder
        x = self.up1(bottleneck, skip8)
        x = self.up2(x,          skip4)
        x = self.up3(x,          skip2)
        x = self.up4(x,          skip1)
        return self.outc(x)

class TriPathDC(nn.Module):
    """
    Three-encoder UNet:
      - Shared stem (DoubleConv in_ch -> b) !!!Fused
      - Encoder paths:
          CNN path   : DoubleConv blocks
          DCN path   : DCNBlock blocks (your deformable impl)
          DSConv path: DSConvBlockMMCV blocks (new)
      - Gated fusion at each skip scale + bottleneck
      - Standard UNet decoder
    Output: logits with num_classes channels (use sigmoid for multilabel)
    """
    def __init__(self, in_ch=3, num_classes=2, base_ch=64):
        super().__init__()
        b = base_ch

        # Shared stem (full-res skip)
        self.stem = DoubleConv(in_ch, b)  # -> x1_shared

        # -------- Encoders (three branches) --------
        # Stage H/2:  b -> 2b
        self.down1_cnn = Down(b,  2*b, block_cls=DCNBlock)
        self.down1_dcn = Down(b,  2*b, block_cls=DCNBlock)
        self.down1_dsc = Down(b,  2*b, block_cls=DCNBlock)   # <-- was NSConvBlock

        # Stage H/4:  2b -> 4b
        self.down2_cnn = Down(2*b, 4*b, block_cls=DCNBlock)
        self.down2_dcn = Down(2*b, 4*b, block_cls=DCNBlock)
        self.down2_dsc = Down(2*b, 4*b, block_cls=DCNBlock)  # <-- was NSConvBlock

        # Stage H/8:  4b -> 8b
        self.down3_cnn = Down(4*b, 8*b, block_cls=DCNBlock)
        self.down3_dcn = Down(4*b, 8*b, block_cls=DCNBlock)
        self.down3_dsc = Down(4*b, 8*b, block_cls=DCNBlock)  # <-- was NSConvBlock

        # Stage H/16 (bottleneck): 8b -> 16b
        self.down4_cnn = Down(8*b,  16*b, block_cls=DCNBlock)
        self.down4_dcn = Down(8*b,  16*b, block_cls=DCNBlock)
        self.down4_dsc = Down(8*b,  16*b, block_cls=DCNBlock) # <-- was NSConvBlock

        # -------- Fusions (unchanged) --------
        self.fuse_2b  = GateFuse3(in_ch_each=2*b,  out_ch=2*b)   # H/2
        self.fuse_4b  = GateFuse3(in_ch_each=4*b,  out_ch=4*b)   # H/4
        self.fuse_8b  = GateFuse3(in_ch_each=8*b,  out_ch=8*b)   # H/8
        self.fuse_16b = GateFuse3(in_ch_each=16*b, out_ch=16*b)  # bottleneck

        # -------- Decoder (unchanged) --------
        self.up1 = Up(16*b, 8*b,  8*b)  # from fused bottleneck, skip=fused 8b
        self.up2 = Up(8*b,  4*b,  4*b)
        self.up3 = Up(4*b,  2*b,  b)
        self.up4 = Up(b,    b,    b)    # last skip = shared stem
        self.outc = OutConv(b, num_classes)

    def forward(self, x):
        x1_shared = self.stem(x)  # [B,b,H,W]

        # Branch 1: CNN
        x2_c = self.down1_cnn(x1_shared)
        x3_c = self.down2_cnn(x2_c)
        x4_c = self.down3_cnn(x3_c)
        x5_c = self.down4_cnn(x4_c)

        # Branch 2: DCN
        x2_d = self.down1_dcn(x1_shared)
        x3_d = self.down2_dcn(x2_d)
        x4_d = self.down3_dcn(x3_d)
        x5_d = self.down4_dcn(x4_d)

        # Branch 3: DSConv (new)
        x2_s = self.down1_dsc(x1_shared)
        x3_s = self.down2_dsc(x2_s)
        x4_s = self.down3_dsc(x3_s)
        x5_s = self.down4_dsc(x4_s)

        # Fusions
        bottleneck = self.fuse_16b(x5_c, x5_d, x5_s)  # [B,16b,H/16,W/16]
        skip8      = self.fuse_8b (x4_c, x4_d, x4_s)  # [B, 8b,H/8 ,W/8 ]
        skip4      = self.fuse_4b (x3_c, x3_d, x3_s)  # [B, 4b,H/4 ,W/4 ]
        skip2      = self.fuse_2b (x2_c, x2_d, x2_s)  # [B, 2b,H/2 ,W/2 ]
        skip1      = x1_shared

        # Decoder
        x = self.up1(bottleneck, skip8)
        x = self.up2(x,          skip4)
        x = self.up3(x,          skip2)
        x = self.up4(x,          skip1)
        return self.outc(x)
