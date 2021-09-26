import pytest
import torch
from torch.nn.parameter import Parameter

from swem.models.word_drop_embedding import WordDropEmbedding


class TestWordDropEmbedding:
    def test_eval(self):
        emb = WordDropEmbedding(10, 2, p=0.5)
        weight = Parameter(torch.arange(0, 20, dtype=torch.float32).view(10, 2))
        emb.weight = weight
        emb.eval()
        input = torch.arange(0, 10, dtype=torch.int64).view(1, 10)
        output = emb(input)
        assert torch.allclose(weight, output.view(10, 2))

    def test_train(self):
        emb = WordDropEmbedding(10, 2, p=0.8)
        emb.train()
        input = torch.arange(0, 10, dtype=torch.int64).view(1, 10)
        output = emb(input)
        assert output.size() == (1, 10, 2)

    def test_errors(self):
        with pytest.raises(ValueError):
            WordDropEmbedding(2, 2, p=-1)

        with pytest.raises(ValueError):
            WordDropEmbedding(2, 2, p=1)
