import pytest
import torch

from swem.torch_utils import to_device


def test_to_device():
    x = torch.tensor([1, 2])

    output = to_device(
        [x],
        device="cpu",
    )
    assert isinstance(output, list)
    assert len(output) == 1
    assert isinstance(output[0], torch.Tensor)

    output = to_device(
        (x,),
        device="cpu",
    )
    assert isinstance(output, tuple)
    assert len(output) == 1
    assert isinstance(output[0], torch.Tensor)

    output = to_device(
        {"key": x},
        device="cpu",
    )
    assert isinstance(output, dict)
    assert len(output) == 1
    assert isinstance(output["key"], torch.Tensor)

    with pytest.raises(ValueError):
        to_device(
            [x, 1],
            device="cpu",
        )

    output = to_device([1, x], device="cpu", non_tensors="drop")
    assert len(output) == 1
    assert isinstance(output[0], torch.Tensor)

    output = to_device([x, 1], device="cpu", non_tensors="ignore")
    assert len(output) == 2
    assert output[1] == 1
