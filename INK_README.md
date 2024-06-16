Branch name: cascade-msdetr_sparse_token

# v1
## v1.0



# v2
## v2.0
Sparse token scheme based on <hybrid_cascade-v1.0>

## v2.1
Sparse token scheme based on <hybrid_cascade-v2.3>

## v2.2
Sparse token scheme based on <hybrid_cascade-v2.5>

## v2.3
Sparse token scheme based on <hybrid_cascade-v2.6>

## v2.4
Sparse token scheme based on <hybrid_cascade-v2.7>

## v2.5
Sparse token scheme based on <hybrid_cascade-v2.9>

### v2.5.1
Following focus-detr, 
fixed topk ratio (0.3) for all layers ->
(0.5, 0.4, 0.3, 0.3, 0.2, 0.1)

### v2.5.2
Amend the sparse token formulation for v2.5.1


# v3
## v3.0
Sparse token scheme based on <hybrid_cascade-v1.6>

### v3.0.1
Based on v3.0, use cumulative cross attention map.

### v3.0.2
Enhance the sparse token formulation of v3.0. Unselected tokens will not go through FFN.

### v3.0.3
Based on v3.0 and following the idea of v3.0.1, consider previous cross attention map, but in this version:

1. use the max value of all attention maps. 



## v3.1
Based on v3.0, use class score to filter out low-confidence queries, which would not participate the calculation of cross attention map.

### v3.1.1
Instead of filtering by class score, use it to modualte attention weights.

### v3.1.2
Based on v3.1.1, change modulation from multiplication to addition.

## v3.2
Inpired by v3.1.2, we try to use class score directly to select salient tokens, instead of using cross attn map.