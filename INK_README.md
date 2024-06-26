Branch: two_stage_variant

# v1
## v1.0
Use learnable anchor box as the anchor for encoder output proposal.

### v1.0.1
Based on v1.0,
1. Change matching criterion of encoder output from anchor to output box
2. Do not clamp the value of learnable reference box

> todo
1. Use multi-class target rather than binary target

## v1.1
Another implementation of roi align


## v1.2
Add self-attention after roi feature.