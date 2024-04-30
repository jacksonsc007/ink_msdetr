import torch
import torch.nn as nn
from detectron2.layers import Conv2d, ShapeSpec, get_norm
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import copy

class MultiScaleAligner_v1732(nn.Module):
    def __init__(
        self,
        num_levels,
        norm_type,
        in_channels,
        out_channels,
    ):
        super(MultiScaleAligner_v1732, self).__init__()
        self.output_convs = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()
        for lvl in range(1, num_levels):
            fusion_norm = get_norm(norm_type, out_channels)
            fusion_conv = Conv2d(
                    (lvl + 1) *out_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    norm=fusion_norm,
                )
            weight_init.c2_xavier_fill(fusion_conv)
            self.fusion_convs.append(fusion_conv)

            output_norm = get_norm(norm_type, out_channels)
            output_conv = Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm=output_norm,
                )
            weight_init.c2_xavier_fill(output_conv)
            self.output_convs.append(output_conv)
    def forward(
        self,
        multi_lvl_feat_list,
    ):
        results = []
        # prev_features = [ multi_lvl_feat_list[-1] ]
        results.append(multi_lvl_feat_list[-1])

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx in range(len(multi_lvl_feat_list)):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                # top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                # prev_features = self.align_convs[idx - 1](prev_features)
                fusion_features = []
                features = multi_lvl_feat_list[-idx -1]
                fusion_features.append(features)

                prev_features = multi_lvl_feat_list[-idx:]
                for prev_f in prev_features:
                    top_down_feature = F.interpolate(prev_f, size=features.shape[-2:], mode="nearest")
                    fusion_features.append(top_down_feature)
                # top_down_features = F.interpolate(prev_features, size=features.shape[-2:], mode="nearest")
                # trans_features = trans_conv(features)
                fusion_features = torch.cat(fusion_features, dim=1)
                fusion_features = self.fusion_convs[idx - 1](fusion_features)
                fusion_features = self.output_convs[idx - 1](fusion_features)
                results.insert(0, fusion_features) 
        return results