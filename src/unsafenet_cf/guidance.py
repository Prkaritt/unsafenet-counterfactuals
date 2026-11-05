"""Classifier guidance through the CogVideoX VAE decoder, with the graph kept small.

The objective is the classifier's log-probability of the target class, evaluated on pixels
decoded from the current latent estimate. The gradient of that objective with respect to
the latent is what steers denoising.

What is and is not in the backward graph
----------------------------------------
Only the **VAE decoder** and the **classifier** are differentiated. The transformer is not.
Guidance is computed from the predicted clean latent, which is treated as a leaf: the
latent is detached, marked ``requires_grad_``, decoded, scored, and the gradient is taken
there. The transformer's own backward graph is never built, which matters because the
unguided generation already peaked at 36.205 GiB of a 44.42 GiB card.

This is the standard "guidance on x0" formulation. It is not the only one, and a method
that needs gradients through the denoiser itself would have to retain that graph. Nothing
here does.

Frame coverage
--------------
The classifier reads 16 of the clip's 49 frames, so a gradient taken through it is exactly
zero on the other 33. ``frame_offset`` rotates which frames are sampled, so across a
multi-step run every frame receives gradient on some steps. Without that rotation guidance
would edit a third of the clip and leave the rest to the prompt.
"""

from __future__ import annotations

import torch

from . import classifier as C


def decode_latent_to_video(vae, latent: torch.Tensor) -> torch.Tensor:
    """Latent -> (B, T, H, W, 3) float in [0, 1], differentiably.

    ``latent`` is in the scaled space the scheduler works in, so it is divided by the VAE's
    scaling factor before decoding, exactly as the pipeline does.
    """
    scaled = latent / vae.config.scaling_factor
    sample = vae.decode(scaled).sample           # (B, 3, T, H, W) in roughly [-1, 1]
    video = (sample + 1.0) / 2.0
    return video.clamp(0, 1).permute(0, 2, 3, 4, 1)


def rotate_frames(video: torch.Tensor, offset: int, num_frames: int) -> torch.Tensor:
    """Select ``num_frames`` evenly spaced frames starting from a rotating offset."""
    t = video.shape[1]
    if t <= num_frames:
        return video
    base = torch.linspace(0, t - 1, num_frames, device=video.device)
    idx = ((base + offset) % t).round().long()
    return video.index_select(1, idx)


def classifier_score(model, video: torch.Tensor, target_index: int,
                     prep: tuple[int, int, int], offset: int = 0) -> torch.Tensor:
    """Scalar log-probability of the target class for a decoded video."""
    frames, height, width = prep
    selected = rotate_frames(video, offset, frames)
    x = C.prepare(selected, frames, height, width)
    return C.target_logit(model(x.float()), target_index)


# The VAE runs in float16, so a gradient flowing back through it can underflow to zero
# long before it is meaningless. The objective is multiplied by this factor before the
# backward pass and the gradient divided by it afterwards, which is the standard
# loss-scaling trick and changes nothing about the result except its representability.
GRAD_SCALE = 1024.0


# Latent frames decoded per guided step. 5 latent frames decode to 17 pixel frames, one
# more than the classifier's 16, and fit in the memory a full decode did not.
DEFAULT_LATENT_WINDOW = 5


def latent_gradient(vae, model, latent: torch.Tensor, target_index: int,
                    prep: tuple[int, int, int], offset: int = 0,
                    grad_scale: float = GRAD_SCALE,
                    window: int | None = DEFAULT_LATENT_WINDOW):
    """Gradient of the target log-probability with respect to ``latent``.

    Returns (gradient, score, scale_used, (window_start, window_width)). The gradient has
    the shape of the whole latent and is zero outside the decoded window. The latent
    passed in is detached inside, so no
    graph from whatever produced it is retained or extended. The gradient is returned in
    float32 and already unscaled.

    If the scaled gradient still contains non-finite values, the scale is halved and the
    backward pass retried, down to a scale of 1. A gradient that is non-finite at every
    scale is returned as it is, so the caller's finiteness check sees the truth rather
    than a silently substituted zero.
    """
    frames_needed = prep[0]
    total_latent_frames = latent.shape[2]
    if window is None:
        start, width = 0, total_latent_frames
    else:
        # Decoding every latent frame with autograd retained does not fit: a full
        # 13-frame decode asked for 8.90 GiB in a single conv3d on top of 40.66 GiB
        # already in use, and ran out of a 44.42 GiB card. A contiguous window of latent
        # frames decodes to enough pixel frames for the classifier and costs a fraction
        # of the memory. The window slides with the step index, so a run covers the whole
        # clip even though no single step does.
        width = min(window, total_latent_frames)
        span = max(total_latent_frames - width + 1, 1)
        start = offset % span

    scale = grad_scale
    while True:
        full = latent.detach()
        leaf = full[:, :, start:start + width].clone().requires_grad_(True)
        with torch.enable_grad():
            video = decode_latent_to_video(vae, leaf)
            if video.shape[1] < frames_needed:
                # Too few pixel frames for the classifier's sampling; widen and retry.
                if width >= total_latent_frames:
                    raise RuntimeError(
                        f"decoded {video.shape[1]} frames from the whole latent, fewer "
                        f"than the {frames_needed} the classifier needs")
                width = min(width + 1, total_latent_frames)
                start = min(start, max(total_latent_frames - width, 0))
                continue
            score = classifier_score(model, video, target_index, prep, offset=0)
            grad_window = torch.autograd.grad(score * scale, leaf)[0]
        grad_window = grad_window.float() / scale
        if torch.isfinite(grad_window).all() or scale <= 1.0:
            grad = torch.zeros(latent.shape, dtype=torch.float32, device=latent.device)
            grad[:, :, start:start + width] = grad_window
            return grad, score.detach(), scale, (int(start), int(width))
        scale /= 2.0


def guidance_diagnostics(grad: torch.Tensor) -> dict:
    """Facts about a gradient that a gate can be decided on."""
    finite = bool(torch.isfinite(grad).all())
    norm = float(grad.norm()) if finite else float("nan")
    return {
        "finite": finite,
        "norm": norm,
        "nonzero": bool(finite and norm > 0.0),
        "max_abs": float(grad.abs().max()) if finite else float("nan"),
        "fraction_nonzero": float((grad != 0).float().mean()),
        "shape": list(grad.shape),
        "dtype": str(grad.dtype),
    }
