import torch
from torch import nn

from swem.models.pooling import HierarchicalPooling
from swem.models.swem import Swem


def test_swem():
    embedding = nn.Embedding(10, 3)
    pooling = HierarchicalPooling(window_size=3)
    swem = Swem(
        embedding=embedding,
        pooling_layer=pooling,
        pre_pooling_dims=(5, 6, 7),
        post_pooling_dims=(8, 9),
    )

    input = torch.randint(0, 10, (8, 21))
    output = swem(input)
    assert output.size() == (8, 9)

    swem = Swem(
        embedding=embedding,
        pooling_layer=pooling,
        pre_pooling_dims=None,
        post_pooling_dims=None,
    )

    output = swem(input)
    assert output.size() == (8, 3)

    swem = Swem(
        embedding=embedding,
        pooling_layer=pooling,
        pre_pooling_dims=(5,),
        post_pooling_dims=(11,),
    )

    output = swem(input)
    assert output.size() == (8, 11)
