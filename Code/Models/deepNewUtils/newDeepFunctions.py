import torch


def square(x):
    return x * x


def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))


def identity(x):
    return x


def squeeze_final_output_new(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    return x


def squeeze_final_output_3d(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    assert x.size()[3] == 1
    x = x[:, :, :, 0, 0]
    return x


def transpose_time_to_spat(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0, 3, 2, 1)


def transpose_time_to_spat2(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    # x = x.view(1, 991, 7, 6)
    # x = x.view(1, 7, 991, 6)
    return x.permute(0, 2, 1, 3)


def transpose_time_to_spat2_3d(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0, 4, 3, 1, 2)
