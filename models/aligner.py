import torch
import torch.nn as nn
from detectron2.layers import Conv2d, ShapeSpec, get_norm
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import copy

class MultiScaleAligner_v1510(nn.Module):
    def __init__(
        self,
        num_levels,
        norm_type,
        in_channels,
        out_channels,
        num_groups = 4,
    ):
        super(MultiScaleAligner_v1510, self).__init__()
        # self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        self.num_groups = num_groups

        for _ in range(num_levels):
            # lateral_norm = get_norm(norm_type, out_channels)
            output_norm = get_norm(norm_type, out_channels)
            # lateral_conv = Conv2d(
            #         in_channels,
            #         out_channels,
            #         kernel_size=1,
            #         bias=False,
            #         norm=lateral_norm
            #     )
            output_conv = Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    norm=output_norm
                )
            # weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            # self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
        
        self.fusion_convs = nn.ModuleList()
        self.align_convs = nn.ModuleList()
        for _ in range(num_levels - 1):
            fusion_norm = get_norm(norm_type, out_channels)
            fusion_conv = Conv2d(
                    2 * out_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    norm=fusion_norm,
                    groups=num_groups,
                )
            weight_init.c2_xavier_fill(fusion_conv)

            align_norm = get_norm(norm_type, out_channels)
            align_conv = Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    norm=align_norm,
                )
            weight_init.c2_xavier_fill(align_conv)

            self.fusion_convs.append(fusion_conv)
            self.align_convs.append(align_conv)
    def forward(
        self,
        multi_lvl_feat_list,
    ):
        results = []
        prev_features = (multi_lvl_feat_list[-1])
        results.append(self.output_convs[0](prev_features))

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, ( output_conv) in enumerate(
             self.output_convs
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = multi_lvl_feat_list[-idx -1]
                # top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                prev_features = self.align_convs[idx - 1](prev_features)
                top_down_features = F.interpolate(prev_features, size=features.shape[-2:], mode="nearest")
                # lateral_features = lateral_conv(features)
                # TODO: try cat & linear
                bs, _, h_, w_ = features.shape
                top_down_features = top_down_features.view(bs, self.num_groups, -1, h_, w_)
                features = features.view(bs, self.num_groups, -1, h_, w_)
                cat_features = torch.cat([features, top_down_features], dim=2).view(bs, -1, h_, w_)
                prev_features = self.fusion_convs[idx - 1](cat_features)
                results.insert(0, output_conv(prev_features)) 
        return results