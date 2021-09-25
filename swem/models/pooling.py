import torch
from torch import nn


class HierarchicalPooling(nn.Module):
    """Hierarchical Pooling layer (see
    [Baselines need more love](https://arxiv.org/abs/1808.09843)).

    Args:
        window_size (int): Size of the pooling window in the mean pooling step.

    Shape:
        input: (batch_size, seq_len, enc_dim)
        output: (batch_size, enc_dim)
    """

    def __init__(self, window_size: int):
        super().__init__()
        self.avg_pooling = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1)
        self.window_size = window_size

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        seq_len = input.size(1)

        # If the sequenze is particularly short there is not much to do
        if seq_len <= self.window_size:
            return torch.mean(input, dim=1)

        output = self.avg_pooling(input.unsqueeze(1)).squeeze(1)
        output, _ = torch.max(output, dim=-2)
        return output
