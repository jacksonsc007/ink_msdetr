Branch: two_stage_variant

# v1
## v1.0
Use learnable anchor box as the anchor for encoder output proposal.

### v1.0.1
Based on v1.0,
1. Change matching criterion of encoder output from anchor to output box
2. Do not clamp the value of learnable reference box

### v1.0.2
Based on v1.0.1, modify the encoder matcher. 

For the original encoder matcher, anchors are used to evaluate matching cost and `args.set_cost_class` is explicit set as 0.

```python
enc_matcher = HungarianMatcher(cost_class=0, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
```

In this version, we restore the default setting of `args.set_cost_class` for encoder matcher.
```python
enc_matcher = HungarianMatcher(cost_class=args.set_cost_bbox, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
```

> todo
1. Use multi-class target rather than binary target

## v1.1
Another implementation of roi align


## v1.2
Add self-attention after roi feature.