Branch: reppoints

The concept of sampling locations of deformable attention share the same spirit with reppoints. Experiments are conducted to validate if the idea of reppoints could be embedded into the deformable attention process of decoder for detrs.

# v1.0
Following <RepPoints>, we group sampling locations to form pseu-boxes on whice box losses are imposed. 

# v1.1
- [x] pay attention to the initialization of box embed
- [x] apply explicit supervison on the 1st group of reppoints generated from cross-attention.


# v1.1.2
- [x] Don't apply explicit box supervision on the pseudo-boxes of 1st group of reppoints.

Limit the number of reppoints to 8. Reppoints are same for all feature levels.

- [x] Change the output number of box_head for decoder
- [x] box_head output logits now
- [x] attention weight for each points

## todo



