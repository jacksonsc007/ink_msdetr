Branch name: cascade-msdetr-improve_multi_scale_align

Some changes from the original work:
1. resnet18 is the default backbone
2. the default number of queries is 100
3. shorter image size for training and inference is fixes as 480


# v1 - deformable-attention-like MultiScaleSampler
## v1.0
Use multi_scale_sampler (v1.4.1 in branch asymmetric_enc_dec )proposed in branch asymmetric_enc_dec

## v1.1
Use multi_scale_sampler (v1.4 in branch asymmetric_enc_dec )proposed in branch asymmetric_enc_dec

only use 1 head

## v1.2
Based on v1.0, discard FFN.


# v2 FPN-like MultiScaleAligner
## v2.1

Use MultiScaleAligner (v1.7.3.4) 