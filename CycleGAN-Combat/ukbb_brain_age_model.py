"""
PyTorch adaptation of the UK Biobank brain-age model from
UKBB_age_pretrain.ipynb.

Designed to plug into the existing training path:

    model = make_age_model(device)
    pred = model(x)

Expected input:
    x: [B, 1, D, H, W], with default D/H/W = 121/121/145

Returned output:
    pred: [B]

The original notebook uses TensorFlow/Keras with channels-last tensors
[B, D, H, W, 1]. This implementation uses PyTorch channels-first tensors.

Important notebook issue
------------------------
The notebook calls every convolution block with the original `images`
tensor instead of passing the output of one block into the next. Therefore,
its 32-channel and 64-channel branches are discarded, and only the
128-channel block reaches the output.

This file provides:
    architecture="corrected"      Intended sequential 32 -> 64 -> 128 model.
    architecture="notebook_exact" Exact effective architecture of the notebook.

The corrected architecture is the default.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union

import torch
from torch import nn


SpatialShape = Tuple[int, int, int]
ArchitectureName = Literal["corrected", "notebook_exact"]


def _as_spatial_shape(shape: Sequence[int]) -> SpatialShape:
    """Validate and convert a spatial shape to a 3-integer tuple."""
    if len(shape) != 3:
        raise ValueError(
            f"input_shape must contain exactly three spatial dimensions, got {shape}."
        )

    result = tuple(int(v) for v in shape)
    if any(v <= 0 for v in result):
        raise ValueError(f"All input dimensions must be positive, got {result}.")

    return result  # type: ignore[return-value]


def _pool_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """PyTorch/Keras valid-pooling output-size formula."""
    return (
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1)
        // stride
        + 1
    )


def _max_pool_shape(shape: SpatialShape, number_of_pools: int) -> SpatialShape:
    """Apply repeated 2x2x2, stride-2 valid max-pooling symbolically."""
    output = shape
    for _ in range(number_of_pools):
        output = tuple(
            _pool_output_size(size, kernel_size=2, stride=2)
            for size in output
        )
    return output  # type: ignore[return-value]


def _post_average_pool_shape(shape: SpatialShape) -> SpatialShape:
    """Apply the notebook's AveragePool3D((2, 3, 2), strides=2)."""
    kernels = (2, 3, 2)
    return tuple(
        _pool_output_size(size, kernel_size=kernel, stride=2)
        for size, kernel in zip(shape, kernels)
    )  # type: ignore[return-value]


def normalize_each_volume(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Reproduce the notebook's per-volume min-max normalization.

    Each sample is independently transformed to approximately [0, 1]:

        x_normalized = (x - min(x)) / (max(x) - min(x))

    Constant-valued volumes become zero rather than producing NaNs.
    """
    if x.ndim != 5:
        raise ValueError(
            "normalize_each_volume expects [B, C, D, H, W], "
            f"but received shape {tuple(x.shape)}."
        )

    spatial_dims = (2, 3, 4)
    minimum = x.amin(dim=spatial_dims, keepdim=True)
    maximum = x.amax(dim=spatial_dims, keepdim=True)
    value_range = maximum - minimum

    normalized = (x - minimum) / value_range.clamp_min(eps)
    normalized = torch.where(
        value_range > eps,
        normalized,
        torch.zeros_like(normalized),
    )
    return normalized


class UKBBConvolutionBlock(nn.Module):
    """
    PyTorch equivalent of the notebook's convolution_block:

        Conv3D(kernel=3, stride=1, padding="same")
        InstanceNormalization(center=False, scale=False)
        MaxPool3D(kernel=2, stride=2, padding="valid")
        ReLU

    Keras InstanceNormalization(center=False, scale=False) corresponds to
    PyTorch InstanceNorm3d(affine=False).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.InstanceNorm3d(
                out_channels,
                affine=False,
                track_running_stats=False,
            ),
            nn.MaxPool3d(
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UKBBBrainAge3D(nn.Module):
    """
    PyTorch adaptation of the shared UKBB 3D brain-age predictor.

    Parameters
    ----------
    input_shape:
        Spatial shape expected after your existing
        `prepare_batch_for_3dcnn`/generated-image loading code.
        Your current data uses (121, 121, 145).

    architecture:
        "corrected":
            Uses the intended sequential stack:
            1 -> 32 -> 64 -> 128 channels.

        "notebook_exact":
            Reproduces the effective graph in the shared notebook.
            Because of the notebook bug, only 1 -> 128 is used.

    normalize_input:
        Applies the notebook's independent min-max normalization to every
        volume inside forward(), allowing the surrounding data/training path
        to remain unchanged.

    dropout:
        The notebook uses dropout=0.5.

    initial_age_bias:
        Optional initial bias for the final age neuron. The notebook's Keras
        default is zero, so None keeps zero initialization. You may set this
        to the training-set mean age if desired.
    """

    def __init__(
        self,
        input_shape: Sequence[int] = (121, 121, 145),
        architecture: ArchitectureName = "corrected",
        normalize_input: bool = True,
        normalization_eps: float = 1e-6,
        dropout: float = 0.5,
        initial_age_bias: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.input_shape = _as_spatial_shape(input_shape)
        self.architecture = architecture
        self.normalize_input = bool(normalize_input)
        self.normalization_eps = float(normalization_eps)

        if architecture == "corrected":
            self.features = nn.Sequential(
                UKBBConvolutionBlock(1, 32),
                UKBBConvolutionBlock(32, 64),
                UKBBConvolutionBlock(64, 128),
            )
            number_of_max_pools = 3
        elif architecture == "notebook_exact":
            # This is the only convolution branch that reaches the output in
            # the original notebook's actual computation graph.
            self.features = nn.Sequential(
                UKBBConvolutionBlock(1, 128),
            )
            number_of_max_pools = 1
        else:
            raise ValueError(
                "architecture must be either 'corrected' or "
                f"'notebook_exact', got {architecture!r}."
            )

        self.post_conv = nn.Sequential(
            nn.Conv3d(
                128,
                64,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.InstanceNorm3d(
                64,
                affine=False,
                track_running_stats=False,
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(
                kernel_size=(2, 3, 2),
                stride=2,
                padding=0,
            ),
        )

        self.dropout = nn.Dropout(p=float(dropout))
        self.reg_conv = nn.Conv3d(
            64,
            64,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        pooled_shape = _max_pool_shape(
            self.input_shape,
            number_of_pools=number_of_max_pools,
        )
        final_spatial_shape = _post_average_pool_shape(pooled_shape)

        if any(size <= 0 for size in final_spatial_shape):
            raise ValueError(
                f"Input shape {self.input_shape} becomes invalid after pooling. "
                f"Final spatial shape would be {final_spatial_shape}."
            )

        self.final_spatial_shape = final_spatial_shape
        flattened_features = 64
        for size in final_spatial_shape:
            flattened_features *= size

        self.flattened_features = flattened_features
        self.age_value = nn.Linear(flattened_features, 1, bias=True)

        self.reset_parameters(initial_age_bias=initial_age_bias)

    def reset_parameters(self, initial_age_bias: Optional[float] = None) -> None:
        """
        Initialize Conv3d and Linear layers using Glorot/Xavier uniform,
        which is closer to Keras defaults than the old Kaiming loop.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if initial_age_bias is not None:
            nn.init.constant_(self.age_value.bias, float(initial_age_bias))

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept the current channels-first format and, for convenience,
        channels-last tensors as well.
        """
        if not torch.is_floating_point(x):
            x = x.float()

        # Current training path: [B, 1, D, H, W]
        if x.ndim == 5 and x.shape[1] == 1:
            prepared = x

        # Optional compatibility with the original Keras layout:
        # [B, D, H, W, 1] -> [B, 1, D, H, W]
        elif x.ndim == 5 and x.shape[-1] == 1:
            prepared = x.permute(0, 4, 1, 2, 3).contiguous()

        # Also tolerate a missing channel dimension: [B, D, H, W]
        elif x.ndim == 4:
            prepared = x.unsqueeze(1)

        else:
            raise ValueError(
                "Expected input with shape [B, 1, D, H, W], "
                "[B, D, H, W, 1], or [B, D, H, W]. "
                f"Received {tuple(x.shape)}."
            )

        actual_shape = tuple(int(v) for v in prepared.shape[-3:])
        if actual_shape != self.input_shape:
            raise ValueError(
                f"This model was constructed for spatial shape {self.input_shape}, "
                f"but received {actual_shape}. Construct it with "
                f"input_shape={actual_shape} if your data shape is different."
            )

        return prepared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            [B, 1, D, H, W], normally [B, 1, 121, 121, 145].

        Returns
        -------
        torch.Tensor
            Predicted ages with shape [B], matching your current training path.
        """
        x = self._prepare_input(x)

        if self.normalize_input:
            x = normalize_each_volume(x, eps=self.normalization_eps)

        x = self.features(x)
        x = self.post_conv(x)
        x = self.dropout(x)
        x = self.reg_conv(x)
        x = torch.flatten(x, start_dim=1)

        if x.shape[1] != self.flattened_features:
            raise RuntimeError(
                "Unexpected flattened feature size. "
                f"Expected {self.flattened_features}, got {x.shape[1]}."
            )

        x = self.age_value(x)  # [B, 1]
        return x[:, 0]         # [B]


def make_age_model(
    device: Union[torch.device, str],
    input_shape: Sequence[int] = (121, 121, 145),
    architecture: ArchitectureName = "corrected",
    normalize_input: bool = True,
    dropout: float = 0.5,
    initial_age_bias: Optional[float] = None,
) -> UKBBBrainAge3D:
    """
    Drop-in replacement for your current make_age_model function.

    Your existing call can remain exactly:

        make_age_model_fn=lambda: make_age_model(device)

    and the training/evaluation path can remain unchanged.

    The returned model accepts:
        [B, 1, 121, 121, 145]

    and returns:
        [B]
    """
    model = UKBBBrainAge3D(
        input_shape=input_shape,
        architecture=architecture,
        normalize_input=normalize_input,
        dropout=dropout,
        initial_age_bias=initial_age_bias,
    )
    return model.to(device)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    # Lightweight shape check. A smaller shape is used here so this test can
    # run quickly on CPU; the production default remains (121, 121, 145).
    test_shape = (32, 40, 32)
    test_model = make_age_model(
        device="cpu",
        input_shape=test_shape,
        architecture="corrected",
        normalize_input=True,
    )

    test_input = torch.randn(2, 1, *test_shape)
    test_output = test_model(test_input)

    print(test_model)
    print(f"Input shape:  {tuple(test_input.shape)}")
    print(f"Output shape: {tuple(test_output.shape)}")
    print(f"Expected:     {(test_input.shape[0],)}")
    print(f"Trainable parameters: {count_trainable_parameters(test_model):,}")

    assert test_output.shape == (test_input.shape[0],)
    assert torch.isfinite(test_output).all()
    print("Shape and finite-value checks passed.")