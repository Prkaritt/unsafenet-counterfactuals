"""The binary walkway classifier and the differentiable preprocessing in front of it.

One module holds both, on purpose. The preprocessing used at training time and the
preprocessing used inside the guidance loop must be the same function, or the gradient
steers a model that never saw inputs like these. Everything here is differentiable:
``F.interpolate`` and the normalisation are both autograd-friendly, so a gradient taken at
the classifier's output reaches the pixels and, through the VAE, the latent.

Input convention everywhere in this file: a float tensor shaped (B, T, H, W, 3) with values
in [0, 1]. That is what the VAE decoder produces after rescaling, and what a decoded video
looks like before it is turned into frames. ``prepare`` converts it to the (B, 3, T, H, W)
that r3d_18 wants.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Kinetics-400 statistics, from R3D_18_Weights.KINETICS400_V1.transforms().
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)

# Training and guidance both use this geometry. 320x240 keeps the source 3:2 aspect ratio
# of the 720x480 preprocessed clip, so a worker's foot near a painted line stays several
# pixels wide. The Kinetics 112x112 centre crop is deliberately not used: r3d_18 ends in an
# AdaptiveAvgPool3d and accepts any spatial size, verified on 2026-11-04.
DEFAULT_FRAMES = 16
DEFAULT_HEIGHT = 240
DEFAULT_WIDTH = 320


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """r3d_18 with the official Kinetics-400 weights and a fresh binary head."""
    from torchvision.models.video import R3D_18_Weights, r3d_18

    weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def sample_time(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Take ``num_frames`` evenly spaced frames from a (B, T, H, W, C) tensor.

    Indexing, not interpolation, so no frame is a blend of two. Gradients flow to the
    selected frames and are zero elsewhere, which is correct: the classifier genuinely does
    not look at the frames it was not given.
    """
    t = video.shape[1]
    if t == num_frames:
        return video
    idx = torch.linspace(0, t - 1, num_frames, device=video.device).round().long()
    return video.index_select(1, idx)


def prepare(video: torch.Tensor,
            num_frames: int = DEFAULT_FRAMES,
            height: int = DEFAULT_HEIGHT,
            width: int = DEFAULT_WIDTH) -> torch.Tensor:
    """(B, T, H, W, 3) in [0, 1] -> normalised (B, 3, T, h, w) ready for r3d_18.

    Differentiable end to end. Used identically by training, evaluation and guidance.
    """
    if video.dim() != 5 or video.shape[-1] != 3:
        raise ValueError(f"expected (B, T, H, W, 3), got {tuple(video.shape)}")

    x = sample_time(video, num_frames)
    x = x.permute(0, 4, 1, 2, 3)                      # (B, 3, T, H, W)

    b, c, t, h, w = x.shape
    if (h, w) != (height, width):
        x = x.reshape(b * c, t, h, w)                 # fold channels into batch for 2D resize
        x = F.interpolate(x, size=(height, width),
                          mode="bilinear", align_corners=False)
        x = x.reshape(b, c, t, height, width)

    mean = torch.tensor(KINETICS_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(KINETICS_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1, 1)
    return (x - mean) / std


def target_logit(logits: torch.Tensor, target_index: int = 1) -> torch.Tensor:
    """Scalar objective for guidance: the log-probability of the target class.

    Log-softmax rather than the raw logit, so the value cannot be increased by inflating
    every logit together, and the gradient reflects a genuine change in the decision.
    """
    return torch.log_softmax(logits, dim=-1)[:, target_index].sum()
