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
        *args, **kwargs: Same as for the usual nn.Embedding.
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
