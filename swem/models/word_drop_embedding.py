"""Embedding layer with word-drop regularization."""
import torch
from torch import nn


class WordDropEmbedding(nn.Embedding):
    """Embedding layer with word-drop regularization.

    During training drops certain words (token ids) entirely from batches (zeroing
    the corresponding vectors). This layer can be used as a drop-in replacement for
    the usual nn.Embedding.

    Args:
        p (float): Probability with which to drop words (if 0 this layer behaves
          just like a usual embedding layer).
        args/kwargs: Other args and kwargs are handled by nn.Embedding.

    Examples::

        >>> emb = WordDropEmbedding(10, 2, p=0.3)
        >>> input = torch.arange(0, 10, dtype=torch.int64)
        >>> output = emb(input)
        >>> print(output)
        tensor([[-0.8393,  1.3216],
                [-0.3652,  0.2879],
                [ 0.1899,  2.2358],
                [ 1.7776, -1.8437],
                [-0.6406,  0.7939],
                [ 1.0874, -3.2290],
                [ 0.0000, -0.0000],
                [ 0.0000,  0.0000],
                [-0.9666,  1.0100],
                [-0.2444, -2.2618]], grad_fn=<DivBackward0>)

    """

    def __init__(self, *args, p: float, **kwargs):
        super().__init__(*args, **kwargs)
        if p < 0 or p >= 1:
            raise ValueError(
                f"Dropout probability must be non-negative and less than 1, got {p}"
            )
        self.p = p

    def forward(self, input: torch.Tensor) -> torch.FloatTensor:
        output = super().forward(input)
        if self.training and self.p > 0:
            # Compute a mask telling us which ids in the vocab to drop
            vocab_mask = torch.bernoulli(
                torch.Tensor().new_full(
                    (self.num_embeddings,), 1 - self.p, device=input.device
                )
            )

            # The corresponding mask for the input
            batch_mask = torch.index_select(vocab_mask, 0, input.view(-1)).view(
                *input.size(), 1
            )

            # Null the embedded vectors and rescale
            output = (output * batch_mask) / (1 - self.p)

        return output
