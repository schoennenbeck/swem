"""A collection of pooling layers with a common API."""
from typing import Dict, Optional, Union

import torch
from torch import nn


class SwemPoolingLayer(nn.Module):
    """Base class for all pooling layers."""

    def forward(
        self, input: torch.FloatTensor, mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """The pooling computation. This should be overridden by subclasses."""
        raise NotImplementedError()

    @staticmethod
    def from_config(config: Dict[str, Union[str, int]]) -> "SwemPoolingLayer":
        """Construct a pooling layer from the config.

        The config should have a key 'class' specifying the pooling layer to construct,
        e.g. 'AttentionPooling' and additional keys providing the arguments for this
        class's __init__ (e.g. for AttentionPooling 'input_dim').

        Raises:
            NotImplementedError: If the class specified by 'class' is unknown.

        Returns:
            SwemPoolingLayer: A pooling layer defined by the config.
        """
        pooling_type = config["class"]
        if pooling_type == "HierarchicalPooling":
            return HierarchicalPooling(window_size=config["window_size"])
        elif pooling_type == "MaxPooling":
            return MaxPooling()
        elif pooling_type == "MeanPooling":
            return MeanPooling()
        elif pooling_type == "AttentionPooling":
            return AttentionPooling(input_dim=config["input_dim"])
        else:
            raise NotImplementedError(f"Unknown class {pooling_type}")


class HierarchicalPooling(SwemPoolingLayer):
    """Hierarchical Pooling layer (see
    `Baselines need more love <https://arxiv.org/abs/1808.09843>`_ ).

    First mean pooling along the sequence dimension over windows of the given size,
    then max pooling along the sequence dimension.

    The mask input is ignored; it is only there for compatibility.

    Args:
        window_size (int): Size of the pooling window in the mean pooling step.


    Shapes:
        - input: :math:`(\\text{batch_size}, \\text{seq_len}, \\text{enc_dim})`
        - mask: :math:`(\\text{batch_size}, \\text{enc_dim})`
        - output: :math:`(\\text{batch_size}, \\text{enc_dim})`
    """

    def __init__(self, window_size: int):
        super().__init__()
        self.avg_pooling = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1)
        self.window_size = window_size

    @property
    def config(self) -> Dict[str, Union[str, int]]:
        return {"class": "HierarchicalPooling", "window_size": self.window_size}

    def forward(
        self, input: torch.FloatTensor, mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """Pooling forward pass."""

        seq_len = input.size(1)

        # If the sequenze is particularly short there is not much to do
        if seq_len <= self.window_size:
            return torch.mean(input, dim=1)

        output = self.avg_pooling(input.unsqueeze(1)).squeeze(1)
        output, _ = torch.max(output, dim=-2)
        return output


class MeanPooling(SwemPoolingLayer):
    """Simple mean pooling layer.

    Mean pooling along the sequence dimension; ignoring inputs according to the given
    mask.


    Shapes:
        - input: :math:`(\\text{batch_size}, \\text{seq_len}, \\text{enc_dim})`
        - mask: :math:`(\\text{batch_size}, \\text{enc_dim})`
        - output: :math:`(\\text{batch_size}, \\text{enc_dim})`
    """

    @property
    def config(self) -> Dict[str, str]:
        return {"class": "MeanPooling"}

    def forward(
        self, input: torch.FloatTensor, mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """Pooling forward pass."""

        if mask is None:
            return torch.mean(input, dim=-2)
        else:
            masked_input = input * mask.unsqueeze(-1)
            output = torch.sum(masked_input, dim=-2) / torch.sum(
                mask, dim=-1, keepdim=True
            )
            return output


class MaxPooling(SwemPoolingLayer):
    """Simple max pooling layer.

    Max pooling along the sequence dimension.

    The mask input is ignored; it is only there for compatibility.


    Shapes:
        - input: :math:`(\\text{batch_size}, \\text{seq_len}, \\text{enc_dim})`
        - mask: :math:`(\\text{batch_size}, \\text{enc_dim})`
        - output: :math:`(\\text{batch_size}, \\text{enc_dim})`
    """

    @property
    def config(self) -> Dict[str, str]:
        return {"class": "MaxPooling"}

    def forward(
        self, input: torch.FloatTensor, mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """Pooling forward pass."""

        output, _ = torch.max(input, dim=-2)
        return output


class AttentionPooling(SwemPoolingLayer):
    """Pooling with an attention mechanism.

    Pools along the sequence dimension by computing a weighted sum whose weights
    are computed as a small feed forward network (with output size 1) applied to the
    input vectors, followed by a softmax.

    Takes an optional mask telling us which inputs to ignore for the softmax.

    Args:
        input_dim (int): The size of the input vectors.

    Shapes:
        - input: :math:`(\\text{batch_size}, \\text{seq_len}, \\text{enc_dim})`
        - mask: :math:`(\\text{batch_size}, \\text{enc_dim})`
        - output: :math:`(\\text{batch_size}, \\text{enc_dim})`
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention_trafo = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1, bias=False),
        )

    @property
    def config(self) -> Dict[str, Union[int, str]]:
        return {
            "class": "AttentionPooling",
            "input_dim": self.attention_trafo[0].in_features,
        }

    def forward(
        self, input: torch.FloatTensor, mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """Pooling forward pass."""
        attention_logits = self.attention_trafo(input).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        if mask is not None:
            # If we have a mask we null the corresponding attention weights
            # and then renormalize (this has the same effect as only doing
            # the softmax over the non-masked inputs)
            attention_weights = attention_weights * mask
            attention_weights = attention_weights / torch.sum(
                attention_weights, dim=-1, keepdim=True
            )

        return torch.einsum("bsd,bs->bd", input, attention_weights)
