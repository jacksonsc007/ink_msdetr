Branch name: cascade-msdetr-improve-stage_align

> Please refer to msdetr.draw.io for better understanding.

# v1.0
Refer to v1.1 of draw.io drawings. Layernorm is not applied after linear layers.

# v1.1
Based on v1.0, add layernorm after linear layers.

# v1.2
limit the number of previous memory to 1.

# v1.3
Use addition rather than concatenation.

# v1.4
Based on v1.3, add soft mask.