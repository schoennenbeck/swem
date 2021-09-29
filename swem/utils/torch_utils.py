"""Some addtional utilities."""

from typing import Any, Union

import torch


def to_device(
    tensors: Any, device: Union[str, torch.device], non_tensors: str = "error"
) -> Any:
    """Transfer a whole container of tensors to a device.

    The function takes an arbitrarily nested structure of uples, lists, and dicts whose
    entries at some point are tensors (or arbitrary other data types, see argument
    'non_tensors') and returns the same nested structure but with all tensors
    transfered to the given device.

    Args:
        tensors (Any): The container of (eventually) tensors to be transfered.
        device (Union[str, torch.device]): The target device.
        non_tensors (str, optional): A string describing the behaviour of the function
          when a non-tensor is encountered. If 'error' raises a ValueError, if 'ignore'
          the value is returned as is, if 'drop' the value is not included in the output
          (Note that this may subtly change the nested structure of the output since
          lists and tuples may be shorter than in the input and dicts may be missing
          keys). Defaults to "error".

    Raises:
        ValueError: If 'non_tensors' is 'error' and a value not of type list, tuple,
          dict, or tensor is encountered.
        ValueError: If an unsupported option for 'non_tensor' is given.

    Returns:
        Any: Same type and nested structure as the input but with all tensors
        on the given device.

    Examples:
        >>> x = torch.tensor([1.0])
        >>> y = torch.tensor([2.0])
        >>> to_device([x, (x, y), {"x": x, "y": y}], device="cuda:0")
        [tensor([1.], device='cuda:0'),
        (tensor([1.], device='cuda:0'), tensor([2.], device='cuda:0')),
        {'x': tensor([1.], device='cuda:0'), 'y': tensor([2.], device='cuda:0')}]

    """
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, list):
        return [
            t_on_device
            for t in tensors
            if (t_on_device := to_device(t, device=device, non_tensors=non_tensors))
            is not None
        ]
    elif isinstance(tensors, tuple):
        return tuple(
            t_on_device
            for t in tensors
            if (t_on_device := to_device(t, device=device, non_tensors=non_tensors))
            is not None
        )
    elif isinstance(tensors, dict):
        return {
            key: t_on_device
            for key, t in tensors.items()
            if (t_on_device := to_device(t, device=device, non_tensors=non_tensors))
            is not None
        }
    else:
        if non_tensors == "error":
            raise ValueError(f"Encountered non-tensor type {type(tensors)}.")
        elif non_tensors == "ignore":
            return tensors
        elif non_tensors == "drop":
            return None
        else:
            raise ValueError(f"Unknown value for argument 'non_tensors' {non_tensors}")
