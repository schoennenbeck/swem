"""Implementation of the Simple Word Embedding Modell."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn

from swem import __version__
from swem.models.pooling import PoolingConfig, SwemPoolingLayer
from swem.models.word_drop_embedding import EmbeddingConfig, WordDropEmbedding


@dataclass(frozen=True)
class SwemConfig:
    """Configuration for SWEM models."""

    embedding: EmbeddingConfig
    pooling: PoolingConfig
    pre_pooling_dims: tuple[int, ...] | None = None
    post_pooling_dims: tuple[int, ...] | None = None
    dropout: float = 0.2

    def __post_init__(self):
        if isinstance(self.pooling, dict):
            super().__setattr__("pooling", PoolingConfig(**self.pooling))
        if isinstance(self.embedding, dict):
            super().__setattr__("embedding", EmbeddingConfig(**self.embedding))
        if isinstance(self.pre_pooling_dims, list):
            super().__setattr__("pre_pooling_dims", tuple(self.pre_pooling_dims))
        if isinstance(self.post_pooling_dims, list):
            super().__setattr__("post_pooling_dims", tuple(self.post_pooling_dims))

        if self.pre_pooling_dims is not None:
            assert all(
                d > 0 for d in self.pre_pooling_dims
            ), f"Dimension must be greater than 0, got {self.pre_pooling_dims}"
        if self.post_pooling_dims is not None:
            assert all(
                d > 0 for d in self.post_pooling_dims
            ), f"Dimension must be greater than 0, got {self.post_pooling_dims}"
        assert (
            0 <= self.dropout < 1
        ), f"Dropout must be at least 0 and strictly less than 1, got {self.dropout}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(**d)


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
        pre_pooling_dims (tuple[int, ...] | None): Intermediate dimensions for the
          feed forward network applied to the input.
        post_pooling_dims (tuple[int, ...] | None): Intermediate dimensions for the
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
        pre_pooling_dims: tuple[int, ...] | None = None,
        post_pooling_dims: tuple[int, ...] | None = None,
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
    def config(self) -> SwemConfig:
        embedding_config = EmbeddingConfig.from_dict(
            {
                "type": "WordDropEmbedding"
                if isinstance(self.embedding, WordDropEmbedding)
                else "Embedding",
                "embedding_dim": self.embedding.embedding_dim,
                "num_embeddings": self.embedding.num_embeddings,
                "padding_idx": self.embedding.padding_idx,
                "scale_grad_by_freq": self.embedding.scale_grad_by_freq,
                "max_norm": self.embedding.max_norm,
                "norm_type": self.embedding.norm_type,
                "sparse": self.embedding.sparse,
                "p": self.embedding.p
                if isinstance(self.embedding, WordDropEmbedding)
                else None,
            }
        )

        return SwemConfig.from_dict(
            {
                "pre_pooling_dims": self.pre_pooling_dims,
                "post_pooling_dims": self.post_pooling_dims,
                "dropout": self.dropout,
                "pooling": self.pooling_layer.config,
                "embedding": embedding_config,
            }
        )

    @classmethod
    def from_config(cls, config: SwemConfig | dict[str, Any]) -> "Swem":
        """Construct a SWEM-model from a config.

        Instead of a config the user can also provide a dictionary representation of
        the config.

        Args:
            config (SwemConfig | dict[str, Any]): The config to construct the
              model from.

        Examples:
            >>> config = {
            ...     "embedding": {
            ...             "type": "Embedding",
            ...             "num_embeddings": 10,
            ...             "embedding_dim": 2
            ...     },
            ...     "pooling": {
            ...             "type": "HierarchicalPooling",
            ...             "window_size": 5
            ...     },
            ...     "pre_pooling_dims": (5, 5),
            ...     "post_pooling_dims": (6, 6),
            ...     "dropout": 0.1
            ... }
            >>> Swem.from_config(config)
            Swem(
            (embedding): Embedding(10, 2)
            (pooling_layer): HierarchicalPooling(
                (avg_pooling): AvgPool2d(kernel_size=(5, 1), stride=1, padding=0)
            )
            (pre_pooling_trafo): Sequential(
                (0): Linear(in_features=2, out_features=5, bias=True)
                (1): ReLU()
                (2): Dropout(p=0.1, inplace=False)
                (3): Linear(in_features=5, out_features=5, bias=True)
                (4): ReLU()
                (5): Dropout(p=0.1, inplace=False)
            )
            (post_pooling_trafo): Sequential(
                (0): Linear(in_features=5, out_features=6, bias=True)
                (1): ReLU()
                (2): Dropout(p=0.1, inplace=False)
                (3): Linear(in_features=6, out_features=6, bias=True)
            )
            )

        """
        if isinstance(config, dict):
            config = SwemConfig.from_dict(config)

        pooling_layer = SwemPoolingLayer.from_config(config.pooling)

        embedding_config = asdict(config.embedding)

        if embedding_config.pop("type") == "WordDropEmbedding":
            embedding = WordDropEmbedding(**embedding_config)
        else:
            embedding_config.pop("p")
            embedding = nn.Embedding(**embedding_config)

        return cls(
            embedding=embedding,
            pooling_layer=pooling_layer,
            pre_pooling_dims=config.pre_pooling_dims,
            post_pooling_dims=config.post_pooling_dims,
            dropout=config.dropout,
        )

    def save(self, path: str | Path):
        """Save this model to disk.

        The model will be stored to the directory given by the path as two files
        'config.json' containing the model config and 'weights.pt" containing the
        weights of the layers.

        Args:
            path (str | Path): The directory in which to store the model file.
              Should be empty. If the directory does not exist it will be generated (
              this does not apply to intermediate directories).

        Raises:
            FileNotFoundError: If the parent directory of path does not exist.
            NotADirectoryError: If path already exists and is not a directory.
            FileExistsError: If path is a non-empty directory.
        """
        path = Path(path)
        if not path.parent.exists():
            raise FileNotFoundError(
                f"Parent path {path.parent.absolute()} does not exist."
            )

        if path.exists():
            if not path.is_dir():
                raise NotADirectoryError(
                    f"Path {path.absolute()} exists and is not a directory."
                )
            elif any(path.iterdir()):
                raise FileExistsError(
                    f"Path {path.absolute()} exists and is not empty."
                )
        else:
            path.mkdir()

        with open(path / "config.json", "w") as f:
            config = asdict(self.config)
            config["version"] = __version__
            json.dump(config, f)

        torch.save(self.state_dict(), path / "weights.pt")

    @classmethod
    def load(cls, path: str | Path) -> "Swem":
        """Load a model that was previously saved on disk.

        Args:
            path (str | Path): The directory in which contains the config and
              weights for the stored model.

        Raises:
            NotADirectoryError: If the given path is not a directory.

        Returns:
            [Swem]: The saved model.
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"Path {path.absolute()} is not a directory.")

        with open(path / "config.json") as f:
            config = json.load(f)

        version = config.pop("version")
        if version != __version__:
            warnings.warn(
                f"Trying to load model from version {version} while using version {__version__}. This may lead to unintended behaviour."
            )

        model = cls.from_config(config)
        weights = torch.load(path / "weights.pt", map_location="cpu")
        model.load_state_dict(weights)

        return model

    def forward(self, input: torch.Tensor) -> torch.FloatTensor:
        output = self.embedding(input)
        output = self.pre_pooling_trafo(output)
        output = self.pooling_layer(output)
        output = self.post_pooling_trafo(output)
        return output
