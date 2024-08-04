Branch name: shared_content_query

# v1.0
On the basis the ms-detr, infuse additional shared content query, to aggregate information from all objects.

# v1.1
Object queries for decoder now share the same content query.

## v1.1.1
Compute the similarity between query after cross-attention and the learnable query emebed, then use the similarity to weight cross-attention output.