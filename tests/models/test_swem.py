import torch
from torch import nn

from swem.models.pooling import HierarchicalPooling
from swem.models.swem import Swem


class TestSwem:
    embedding = nn.Embedding(10, 3)
    pooling = HierarchicalPooling(window_size=3)

    def test_generic(self):
        swem = Swem(
            embedding=self.embedding,
            pooling_layer=self.pooling,
            pre_pooling_dims=(5, 6, 7),
            post_pooling_dims=(8, 9),
        )

        input = torch.randint(0, 10, (8, 21))
        output = swem(input)
        assert output.size() == (8, 9)

    def test_no_trafos(self):
        swem = Swem(
            embedding=self.embedding,
            pooling_layer=self.pooling,
            pre_pooling_dims=None,
            post_pooling_dims=None,
        )

        input = torch.randint(0, 10, (8, 21))
        output = swem(input)
        assert output.size() == (8, 3)

    def test_short_trafos(self):
        swem = Swem(
            embedding=self.embedding,
            pooling_layer=self.pooling,
            pre_pooling_dims=(5,),
            post_pooling_dims=(11,),
        )

        input = torch.randint(0, 10, (8, 21))
        output = swem(input)
        assert output.size() == (8, 11)
