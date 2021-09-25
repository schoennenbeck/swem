from itertools import chain
from typing import Optional, Tuple

import torch
from torch import nn


class Swem(nn.Module):
    """Simple Word Embedding model (see
    [Baselines need more love](https://arxiv.org/abs/1808.09843)).

    The model consists of an embedding layer, a feed forward network that is applied
    separately to each word vector, a pooling layer that pools the vectors belonging to
    the same text into a single vector, and another feed forward network that is applied
    to this pooled vector.

    Args:
        embedding (nn.Embedding): The embedding layer used by the model.
        pooling_layer (nn.Module): The pooling layer to be used by the model.
        pre_pooling_dims (Optional[Tuple[int, ...]]): Intermediate dimensions for the
        feed forward network applied to the input.
        post_pooling_dims (Optional[Tuple[int, ...]]): Intermediate dimensions for the
        feed forward network applied to the output of the pooling layer.
        dropout (float): Dropout probability after each layer in both feed forward
        subnetworks.

    Shapes:
        Input: (batch_size, seq_len, dim)
        Output: (batch_size, dim)
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        pooling_layer: nn.Module,
        pre_pooling_dims: Optional[Tuple[int, ...]] = None,
        post_pooling_dims: Optional[Tuple[int, ...]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = embedding
        self.pooling_layer = pooling_layer
        if pre_pooling_dims is None:
            self.pre_pooling_trafo = nn.Identity()
        else:
            pre_dims = [embedding.embedding_dim, *pre_pooling_dims]
            self.pre_pooling_trafo = nn.Sequential(
                *chain(
                    *[
                        (nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Dropout(dropout))
                        for dim_in, dim_out in zip(pre_dims[:-1], pre_dims[1:])
                    ]
                )
            )

        if post_pooling_dims is None:
            self.post_pooling_trafo = nn.Identity()
        else:
            pooling_dim = (
                embedding.embedding_dim
                if pre_pooling_dims is None
                else pre_pooling_dims[-1]
            )
            if len(post_pooling_dims) == 1:
                self.post_pooling_trafo = nn.Linear(pooling_dim, post_pooling_dims[0])
            else:
                post_dims = [pooling_dim, *post_pooling_dims[:-1]]
                self.post_pooling_trafo = nn.Sequential(
                    *chain(
                        *[
                            (nn.Linear(dim_in, dim_out), nn.ReLU(), nn.Dropout(dropout))
                            for dim_in, dim_out in zip(post_dims[:-1], post_dims[1:])
                        ]
                    ),
                    torch.nn.Linear(post_dims[-1], post_pooling_dims[-1]),
                )

    def forward(self, input: torch.Tensor) -> torch.FloatTensor:
        output = self.embedding(input)
        output = self.pre_pooling_trafo(output)
        output = self.pooling_layer(output)
        output = self.post_pooling_trafo(output)
        return output
