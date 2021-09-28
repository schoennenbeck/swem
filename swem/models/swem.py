"""Implementation of the Simple Word Embedding Modell."""

from itertools import chain
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from swem.models.pooling import SwemPoolingLayer
from swem.models.word_drop_embedding import WordDropEmbedding


class Swem(nn.Module):
    """Simple Word Embedding model (see
    `Baselines need more love <https://arxiv.org/abs/1808.09843>`_ ).

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
        - input: :math:`(\\text{batch_size}, \\text{seq_len})`
        - output: :math:`(\\text{batch_size}, \\text{enc_dim})`, where
          :math:`\\text{enc_dim}` is the last of the post_pooling_dims (if given,
          otherwise the last of the pre_pooling_dims or failing that the
          embedding_dimension).
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        pooling_layer: SwemPoolingLayer,
        pre_pooling_dims: Optional[Tuple[int, ...]] = None,
        post_pooling_dims: Optional[Tuple[int, ...]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = embedding
        self.pooling_layer = pooling_layer
        self.pre_pooling_dims = pre_pooling_dims
        self.post_pooling_dims = post_pooling_dims
        self.dropout = dropout

        if pre_pooling_dims is None:
            self.pre_pooling_trafo: nn.Module = nn.Identity()
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
            self.post_pooling_trafo: nn.Module = nn.Identity()
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

    @property
    def config(self) -> Dict[str, Any]:
        embedding_config = {
            "class": "WordDropEmbedding"
            if isinstance(self.embedding, WordDropEmbedding)
            else "Embedding",
            "embedding_dim": self.embedding.embedding_dim,
            "num_embeddings": self.embedding.num_embeddings,
            "padding_idx": self.embedding.padding_idx,
            "scale_grad_by_freq": self.embedding.scale_grad_by_freq,
            "max_norm": self.embedding.max_norm,
            "norm_type": self.embedding.norm_type,
            "sparse": self.embedding.sparse,
        }

        if isinstance(self.embedding, WordDropEmbedding):
            embedding_config["p"] = self.embedding.p

        return {
            "pre_pooling_dims": self.pre_pooling_dims,
            "post_pooling_dims": self.post_pooling_dims,
            "dropout": self.dropout,
            "pooling": self.pooling_layer.config,
            "embedding": embedding_config,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Swem":
        """Construct a SWEM-model from a config.

        The config should have the following keys:

            - pooling: A config for a SwemPoolingLayer.
            - embedding: A dict with key 'class' specifying either 'Embedding' or
              'WordDropEmbedding' and further keys according to the arguments of the
              correct class's __init__.
            - pre_pooling_dims: Tuple of int as for __init__.
            - post_pooling_dims: Tuple of int as for __init__.
            - dropout: float as for __init__.

        Args:
            config (Dict[str, Any]): The config to construct the model from.

        """
        pooling_config = config.pop("pooling")
        pooling_layer = SwemPoolingLayer.from_config(pooling_config)

        embedding_config = config.pop("embedding")
        if embedding_config.pop("class") == "WordDropEmbedding":
            embedding = WordDropEmbedding(**embedding_config)
        else:
            embedding = nn.Embedding(**embedding_config)

        return cls(embedding=embedding, pooling_layer=pooling_layer, **config)

    def forward(self, input: torch.Tensor) -> torch.FloatTensor:
        output = self.embedding(input)
        output = self.pre_pooling_trafo(output)
        output = self.pooling_layer(output)
        output = self.post_pooling_trafo(output)
        return output
