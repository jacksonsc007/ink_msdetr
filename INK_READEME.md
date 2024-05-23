Branch name: asymmetric_enc_dec

We apply a 6-layer encoder and 7-layer decoder transformer structure. We hope that the additional decoder layer could
help eliminate unnecessary tokens, which participate in the encoder.

> Please refer to msdetr.drawio / asymmetric_enc_dec for more details.
# v1.0
Naive design.


# v1.1
Use stage aligner.