from .Pipelines import lightning_training_wrapper
from .Datasets import TipletDataset

from typing import Any, Literal
import numpy
import torch


def _mahalanobis_whitening(data: torch.Tensor | numpy.ndarray, should_normalize: bool = True) -> torch.Tensor | numpy.ndarray:
    r"""
    Perform ZCA whitening on a single image.

    :math:`d^2(x) = (x-\mu)^T\sum^{-1}(x-\mu)`

    Where :math:`\sum` is Identity.

    :param data: Matrix representing single image to normalize.
    :param should_normalize: Set to `True` if image is not scaled and normalized.
    :return: Whitened image.
    """
    x = data

    if should_normalize:
        x = x - data.mean()
        x = x / data.std()

    #cov = numpy.cov(x, rowvar=x.shape[0] > 1)
    cov = torch.cov(x)
    #u, s, v = numpy.linalg.svd(cov)
    u, s, v = torch.linalg.svd(cov)
    s += 1e-8

    #whitening = numpy.dot(u,
    #                      numpy.dot(
    #                          numpy.diag(
    #                              1.0 / numpy.sqrt(s)
    #                          ),
    #                          u.T
    #                      ))

    whitening = torch.mm(
        u,
        torch.mm(
            torch.diag(1.0 / torch.sqrt(s)),
            u.T
        )
    )

    #return numpy.dot(whitening, x)
    return torch.mm(whitening, x)


def mahalanobis_whitening_batch(data: torch.Tensor, should_normalize: bool = True,
                                batch_format: Literal["BCWH", "BWHC"] = "BCWH",
                                independent_channels: bool = True) -> torch.Tensor:
    r"""
    Perform ZCA whitening on an image batch on a per-channel basis.
    Assumes batch has only dimensions (Batch, Channel, Width, Height) in the order specified in parameter `format`.
    If `independent_channels` then all the channels of output will be the same.

    :math:`d^2(x) = (x-\mu)^T\sum^{-1}(x-\mu)`

    Where :math:`\sum` is Identity.

    You should make sure the scale of the output is correct with your hypothesis as aggregating channels might have
    unexpected effects.

    :param data: Matrix representing single image to normalize.
    :param should_normalize: Set to `True` if image is not scaled and normalized.
    :param batch_format: Dimension format of data.
    :param independent_channels: Treat each channel of each image as independent or aggregate them for normalization.
    :return: Whitened image batch of shape (B*C, H*W).
    """
    assert data.ndim == 4, "Invalid tensor shape, expected (B, C, W, H) but found {} dimensions.".format(data.ndim)
    x = data.float()

    if should_normalize:
        x = x - x.mean()
        x = x / data.float().std()

    if independent_channels:
        if batch_format == "BWHC" or batch_format == "BHWC":
            x = x.permute(0, 3, 1, 2)

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
    else:
        if batch_format == "BCWH" or batch_format == "BCHW":
            x = x.permute(0, 2, 3, 1)

        x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])

    output = _mahalanobis_whitening(x, should_normalize=False)

    if independent_channels:
        # shape here is (C*B, W*H) which is opposite of expected
        if batch_format == "BCWH" or batch_format == "BCHW":
            output = output.reshape(*data.shape)
        else:
            output = output.reshape(data.shape[0], data.shape[3], data.shape[1], data.shape[2])
            output = output.permute(0, 2, 3, 1)
    else:
        # shape here is (B*W*H, C)
        if batch_format == "BCWH" or batch_format == "BCHW":
            output = output.reshape(data.shape[0], data.shape[2], data.shape[3], data.shape[1])  # B W H C
            output = output.permute(0, 3, 1, 2)
        else:
            output = output.reshape(*data.shape)

    return output
