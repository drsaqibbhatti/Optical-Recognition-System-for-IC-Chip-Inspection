# dbnet_convnextv2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# import ConvNeXtV2 variants and their channel dimensions
from backbone.convNextV2Block import (
    convnextv2_nano,
    convnextv2_small,
    convnextv2_medium,
    convnextv2_large,
    convnextv2_nano_dims,
    convnextv2_small_dims,
    convnextv2_medium_dims,
    convnextv2_large_dims,
)

class FPN(nn.Module):
    """FPN that fuses [c2,c3,c4,c5] into a 1/4 map."""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.smooth  = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])
        self.reduce  = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, feats):
        # feats = [c2,c3,c4,c5] low->high
        lat = [l(f) for l, f in zip(self.lateral, feats)]

        # top-down
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:], mode="nearest")

        outs = [s(x) for s, x in zip(self.smooth, lat)]

        # fuse all to c2 resolution
        H, W = outs[0].shape[-2:]
        outs_up = [
            outs[0],
            F.interpolate(outs[1], (H, W), mode="nearest"),
            F.interpolate(outs[2], (H, W), mode="nearest"),
            F.interpolate(outs[3], (H, W), mode="nearest"),
        ]
        fused = torch.cat(outs_up, dim=1)  # [B, 4*out_channels, H, W]
        fused = self.reduce(fused)         # [B, out_channels, H, W]
        return fused


class DBHead(nn.Module):
    """DB head (shrink + thresh, and binary during training)."""
    def __init__(self, in_channels=256, k=50):
        super().__init__()
        self.k = k

        def branch():
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            )

        self.binarize_logits = branch()
        self.thresh_logits = branch()

    def step_function(self, x, y):
        return torch.sigmoid(self.k * (x - y))

    def forward(self, x, is_train: bool):
        shrink_logits = self.binarize_logits(x)
        thresh_logits = self.thresh_logits(x)
        shrink = torch.sigmoid(shrink_logits)
        thresh = torch.sigmoid(thresh_logits)
        out = {
            "shrink": shrink,
            "thresh": thresh,
            "shrink_logits": shrink_logits,
            "thresh_logits": thresh_logits,
        }
        if is_train:
            out["binary"] = self.step_function(shrink, thresh)
        return out


class DBNetConvNeXtV2(nn.Module):
    """
    ConvNeXtV2 backbone -> FPN -> DBHead
    Output maps resized to input H,W.
    """
    def __init__(self, backbone, in_channels_list, out_channels=256, k=50):
        super().__init__()
        self.backbone = backbone
        self.fpn = FPN(in_channels_list, out_channels=out_channels)
        self.head = DBHead(in_channels=out_channels, k=k)

    def forward(self, x):
        H, W = x.shape[-2:]

        feats = self.backbone(x, return_feats=True)  # [c2,c3,c4,c5]
        fused = self.fpn(feats)
        out = self.head(fused, is_train=self.training)

        # force output size to match input
        for key in out:
            out[key] = F.interpolate(out[key], size=(H, W), mode="bilinear", align_corners=False)

        return out


def Nano_detection_model() -> DBNetConvNeXtV2:
    backbone = convnextv2_nano()
    return DBNetConvNeXtV2(
        backbone=backbone,
        in_channels_list=list(convnextv2_nano_dims),
        out_channels=256,
        k=50,
    )


def Small_detection_model() -> DBNetConvNeXtV2:
    backbone = convnextv2_small()
    return DBNetConvNeXtV2(
        backbone=backbone,
        in_channels_list=list(convnextv2_small_dims),
        out_channels=256,
        k=50,
    )


def Medium_detection_model() -> DBNetConvNeXtV2:
    backbone = convnextv2_medium()
    return DBNetConvNeXtV2(
        backbone=backbone,
        in_channels_list=list(convnextv2_medium_dims),
        out_channels=256,
        k=50,
    )


def Large_detection_model() -> DBNetConvNeXtV2:
    backbone = convnextv2_large()
    return DBNetConvNeXtV2(
        backbone=backbone,
        in_channels_list=list(convnextv2_large_dims),
        out_channels=256,
        k=50,
    )
