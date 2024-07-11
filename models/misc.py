import torch
from util import box_ops
def points2box(points):
    """
    Convert representative points to boxes.
    args:
        points: Tensor of shape (bs, num_queries, n_heads, n_lvls, n_points, 2)
            The value is normalized with repect to padded image size.

        valid_ratios: Tensor of shape (bs, n_lvls, 2)
            convert coordinates between padded image and real image
    """
    # convert form padded image coordinates to real image coordinates
    bs, num_queries, n_heads, n_lvls, n_points, _ = points.shape
    points = points.reshape(bs, num_queries, -1, 2)
    points_x, points_y = points.unbind(-1)
    lx = points_x.min(dim=2)[0]
    rx = points_x.max(dim=2)[0]
    ly = points_y.min(dim=2)[0]
    ry = points_y.max(dim=2)[0]
    box = torch.stack([lx, ly, rx, ry], dim=-1)
    box = box_ops.box_xyxy_to_cxcywh(box)
    return box