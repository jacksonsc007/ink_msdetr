Branch name: contrastive_loss_exten


5th try:
Use linear+relu for projection.

6th try:
Use linear+relu+linear for projection.

7th try:
Based on 5th try change the constrast loss formulation.

8th try:
Based on 5th try, modify the value of scale from 1 to 0.1 .

9th try:
Based on 6th try, modify the value of scale from 1 to 0.1 .

10th try:
Based on 6th trym modify the contrast loss formulation. Do not detach gradient.
```python
        obj_embed = self.projector1(query_embeds)
        hs = self.projector2(hs)
```

11th try:
Based on 10th, the contrast loss is computed among all images within the batch, which means that negatives from other images also serves as *contrast samples* for the current image's certain positive sample. 