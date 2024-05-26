Branch name: asymmetric_enc_dec

We apply a 6-layer encoder and 7-layer decoder transformer structure. We hope that the additional decoder layer could
help eliminate unnecessary tokens, which participate in the encoder.

> Please refer to msdetr.drawio / asymmetric_enc_dec for more details.
# v1.0
Naive design.


# v1.1
Use stage aligner

# v1.2
Basd on v1.0, add FPN after the backbone memory.

## v1.2.1
Based on v1.2, Feed positional emebedding before FFN.

# v1.3
Based on v1.0, add multi_scale_alinger of version 1.7.3.2

# v1.4
Use deformable_attention-like MultiScaleSampler.

## v1.4.1
Increase the number of sampling points to 4 by using sampling offsets.