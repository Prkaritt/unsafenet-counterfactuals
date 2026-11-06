"""Registration and mask-restricted compositing for the panel-cover intervention.

The intervention this module supports is narrow and should be stated plainly, because it is
easy to overclaim. A donor clip shows the same machine with its panel cover **closed**. A
source clip shows it **open**. The intervention replaces the pixels of the panel region in
the source with the donor's pixels for that region, registered to the source's own geometry.
It produces a clip in which the cover is closed throughout. It does **not** generate a door
closing: no motion is synthesised, and nothing here models a hinge.

Three things are measured rather than assumed, because each one is a way for this to go
quietly wrong:

* **Registration.** Both cameras are fixed, so a donor from the same view should already sit
  within a pixel or two of the source. "Should" is not a measurement. ``patch_offsets``
  finds the integer shift of several background patches by normalised cross-correlation, and
  those shifts are the landmark correspondences a transform is fitted to. The per-landmark
  reprojection error comes back with the fit and belongs in the record.
* **What is replaced.** Compositing is restricted to an explicit mask. Everything outside it
  is the source's own pixels, bit for bit, which ``unchanged_outside_mask`` verifies.
* **The seam.** ``seam_contrast`` measures the step across the mask boundary against the
  step across an equivalent boundary drawn in untouched background, so a visible seam is a
  number and not an impression.

Numpy only. scipy is not installed in this project's environment, and adding it to the
baseline environment for three functions is not worth the risk to a working CogVideoX stack.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------
# Landmark correspondences by normalised cross-correlation
# ---------------------------------------------------------------------------------


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two equal-shaped patches, in [-1, 1].

    Returns 0.0 for a flat patch, where correlation is undefined, rather than raising. A
    flat patch carries no positional information, so scoring it 0 makes it lose to any
    textured patch, which is the behaviour the caller wants.
    """
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / n) if n > 1e-9 else 0.0


def patch_offsets(src: np.ndarray, dst: np.ndarray,
                  patches: list[tuple[int, int, int, int]],
                  search: int = 12) -> list[dict]:
    """Integer shift of each patch of ``src`` that best matches ``dst``.

    ``patches`` are (x, y, w, h) boxes in ``src``, which should lie on static scene
    structure and **not** on the region to be replaced. Each result carries the offset, the
    correlation peak, and the margin over the second-best offset. A low peak or a thin
    margin means the patch did not localise, and the caller should drop it rather than fit
    to it.
    """
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    h_img, w_img = src.shape[:2]
    out: list[dict] = []
    for (x, y, w, h) in patches:
        if x < 0 or y < 0 or x + w > w_img or y + h > h_img:
            raise ValueError(f"patch {(x, y, w, h)} falls outside {w_img}x{h_img}")
        template = src[y:y + h, x:x + w]
        scores: list[tuple[float, int, int]] = []
        for dy in range(-search, search + 1):
            for dx in range(-search, search + 1):
                yy, xx = y + dy, x + dx
                if yy < 0 or xx < 0 or yy + h > h_img or xx + w > w_img:
                    continue
                scores.append((ncc(template, dst[yy:yy + h, xx:xx + w]), dx, dy))
        if not scores:
            out.append({"patch": [x, y, w, h], "ok": False,
                        "reason": "no in-bounds offset in the search window"})
            continue
        scores.sort(key=lambda s: -s[0])
        best, dx, dy = scores[0]
        # Second best at a different offset, to show the peak is not a plateau.
        runner = next((s[0] for s in scores[1:] if (s[1], s[2]) != (dx, dy)), best)
        out.append({
            "patch": [x, y, w, h],
            "ok": True,
            "offset_xy": [int(dx), int(dy)],
            "ncc": round(best, 5),
            "margin_over_second": round(best - runner, 5),
            "centre_src_xy": [x + w / 2.0, y + h / 2.0],
            "centre_dst_xy": [x + w / 2.0 + dx, y + h / 2.0 + dy],
        })
    return out


# ---------------------------------------------------------------------------------
# Transform fitting
# ---------------------------------------------------------------------------------


def _as_pts(p) -> np.ndarray:
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {arr.shape}")
    return arr


def estimate_translation(src, dst) -> dict:
    """Least-squares pure translation. Needs one correspondence."""
    s, d = _as_pts(src), _as_pts(dst)
    if len(s) != len(d) or len(s) < 1:
        raise ValueError("need at least one matched pair of points")
    t = (d - s).mean(axis=0)
    m = np.array([[1.0, 0.0, t[0]], [0.0, 1.0, t[1]], [0.0, 0.0, 1.0]])
    return _with_residuals("translation", m, s, d, min_points=1)


def estimate_similarity(src, dst) -> dict:
    """Least-squares similarity: uniform scale, rotation, translation. Needs 2 points."""
    s, d = _as_pts(src), _as_pts(dst)
    if len(s) != len(d) or len(s) < 2:
        raise ValueError("a similarity fit needs at least two matched pairs")
    # [x -y 1 0; y x 0 1] @ [a b tx ty] = [u v]
    rows = np.zeros((2 * len(s), 4))
    rhs = np.zeros(2 * len(s))
    rows[0::2] = np.column_stack([s[:, 0], -s[:, 1], np.ones(len(s)), np.zeros(len(s))])
    rows[1::2] = np.column_stack([s[:, 1], s[:, 0], np.zeros(len(s)), np.ones(len(s))])
    rhs[0::2], rhs[1::2] = d[:, 0], d[:, 1]
    (a, b, tx, ty), *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    m = np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])
    return _with_residuals("similarity", m, s, d, min_points=2)


def estimate_affine(src, dst) -> dict:
    """Least-squares affine. Needs 3 points."""
    s, d = _as_pts(src), _as_pts(dst)
    if len(s) != len(d) or len(s) < 3:
        raise ValueError("an affine fit needs at least three matched pairs")
    a = np.column_stack([s, np.ones(len(s))])
    sol, *_ = np.linalg.lstsq(a, d, rcond=None)      # (3, 2)
    m = np.vstack([sol.T, [0.0, 0.0, 1.0]])
    return _with_residuals("affine", m, s, d, min_points=3)


def estimate_homography(src, dst) -> dict:
    """Normalised DLT homography. Needs 4 points.

    Points are isotropically normalised before the SVD and the result is de-normalised, so
    the fit does not depend on the pixel scale the landmarks happen to live at.
    """
    s, d = _as_pts(src), _as_pts(dst)
    if len(s) != len(d) or len(s) < 4:
        raise ValueError("a homography fit needs at least four matched pairs")
    ns, ts = _normalise(s)
    nd, td = _normalise(d)
    rows = []
    for (x, y), (u, v) in zip(ns, nd):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    m = np.linalg.inv(td) @ h @ ts
    if abs(m[2, 2]) > 1e-12:
        m = m / m[2, 2]
    return _with_residuals("homography", m, s, d, min_points=4)


def _normalise(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    scale = np.sqrt(2.0) / d if d > 1e-9 else 1.0
    t = np.array([[scale, 0, -scale * c[0]], [0, scale, -scale * c[1]], [0, 0, 1.0]])
    hom = np.column_stack([pts, np.ones(len(pts))]) @ t.T
    return hom[:, :2], t


def apply_transform(m: np.ndarray, pts) -> np.ndarray:
    p = _as_pts(pts)
    hom = np.column_stack([p, np.ones(len(p))]) @ np.asarray(m, dtype=np.float64).T
    w = hom[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return hom[:, :2] / w


def _with_residuals(kind: str, m: np.ndarray, s: np.ndarray, d: np.ndarray,
                    min_points: int) -> dict:
    pred = apply_transform(m, s)
    err = np.linalg.norm(pred - d, axis=1)
    return {
        "kind": kind,
        "matrix": [[round(float(v), 8) for v in row] for row in m],
        "_matrix": m,
        "n_points": int(len(s)),
        "min_points": min_points,
        "degrees_of_freedom": int(len(s) - min_points),
        "residual_px": {
            "per_point": [round(float(e), 4) for e in err],
            "mean": round(float(err.mean()), 4),
            "max": round(float(err.max()), 4),
            "rms": round(float(np.sqrt((err ** 2).mean())), 4),
        },
        "_residual_note": ("With exactly min_points the fit is exact and the residual is "
                           "zero by construction, which is not evidence of accuracy. Only "
                           "a positive degrees_of_freedom makes the residual informative."),
    }


# ---------------------------------------------------------------------------------
# Warping and compositing
# ---------------------------------------------------------------------------------


def warp(img: np.ndarray, m: np.ndarray, out_shape: tuple[int, int] | None = None
         ) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly resample ``img`` through the transform ``m``.

    ``m`` maps source coordinates to destination coordinates, and the implementation inverts
    it to sample. Returns the warped image and a boolean validity mask, True where the
    sample came from inside the source. Out-of-bounds pixels are zero and must be treated as
    missing data, never as black scene content.
    """
    m = np.asarray(m, dtype=np.float64)
    h_out, w_out = out_shape if out_shape else img.shape[:2]
    inv = np.linalg.inv(m)
    yy, xx = np.meshgrid(np.arange(h_out), np.arange(w_out), indexing="ij")
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    src = apply_transform(inv, flat)
    sx, sy = src[:, 0].reshape(h_out, w_out), src[:, 1].reshape(h_out, w_out)

    h_in, w_in = img.shape[:2]
    valid = (sx >= 0) & (sx <= w_in - 1) & (sy >= 0) & (sy <= h_in - 1)
    x0 = np.clip(np.floor(sx), 0, w_in - 1).astype(int)
    y0 = np.clip(np.floor(sy), 0, h_in - 1).astype(int)
    x1 = np.clip(x0 + 1, 0, w_in - 1)
    y1 = np.clip(y0 + 1, 0, h_in - 1)
    fx = (sx - x0)[..., None] if img.ndim == 3 else (sx - x0)
    fy = (sy - y0)[..., None] if img.ndim == 3 else (sy - y0)

    src_f = img.astype(np.float64)
    top = src_f[y0, x0] * (1 - fx) + src_f[y0, x1] * fx
    bot = src_f[y1, x0] * (1 - fx) + src_f[y1, x1] * fx
    out = top * (1 - fy) + bot * fy
    out = np.where(valid[..., None] if img.ndim == 3 else valid, out, 0.0)
    return out, valid


def box_mean(plane: np.ndarray, k: int) -> np.ndarray:
    """Separable box mean with edge padding; k is forced odd and at least 1."""
    k = max(1, int(k) | 1)
    r = k // 2
    pad = np.pad(plane.astype(np.float64), r, mode="edge")
    acc = np.cumsum(pad, axis=0)
    acc = np.vstack([np.zeros((1, acc.shape[1])), acc])
    mid = (acc[k:, :] - acc[:-k, :]) / k
    acc = np.cumsum(mid, axis=1)
    acc = np.hstack([np.zeros((acc.shape[0], 1)), acc])
    return (acc[:, k:] - acc[:, :-k]) / k


def feather(mask: np.ndarray, radius: int = 0) -> np.ndarray:
    """Turn a boolean mask into a float alpha in [0, 1], softened by ``radius``.

    radius 0 returns a hard 0/1 alpha. A soft edge hides a registration error of a pixel or
    two, which is honest smoothing of a seam; it does not hide a misregistration of ten
    pixels, and it must never be increased until a bad alignment stops looking bad.
    """
    a = mask.astype(np.float64)
    return a if radius <= 0 else np.clip(box_mean(a, 2 * int(radius) + 1), 0.0, 1.0)


def composite(base: np.ndarray, donor: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Alpha-blend ``donor`` over ``base``. Shapes must match; returns uint8."""
    base_f = base.astype(np.float64)
    donor_f = donor.astype(np.float64)
    if base_f.shape != donor_f.shape:
        raise ValueError(f"base {base_f.shape} and donor {donor_f.shape} differ")
    a = alpha[..., None] if base_f.ndim == 3 and alpha.ndim == 2 else alpha
    return np.clip(base_f * (1.0 - a) + donor_f * a, 0, 255).astype(np.uint8)


def unchanged_outside_mask(base: np.ndarray, out: np.ndarray,
                           mask: np.ndarray) -> dict:
    """Check that compositing touched nothing outside ``mask``.

    Returns the count and the largest absolute change outside the mask. For a hard-edged
    composite both must be zero; a feathered alpha legitimately changes a band around the
    mask, so the caller passes the dilated mask it actually blended with.
    """
    outside = ~mask.astype(bool)
    if base.ndim == 3:
        outside = np.repeat(outside[..., None], base.shape[2], axis=2)
    diff = np.abs(base.astype(np.int32) - out.astype(np.int32))[outside]
    changed = int((diff > 0).sum())
    return {
        "pixels_outside_mask": int(outside.sum()),
        "pixels_changed_outside_mask": changed,
        "max_abs_change_outside_mask": int(diff.max()) if diff.size else 0,
        "clean": changed == 0,
    }


def ring(mask: np.ndarray, width: int = 3, outside: bool = True) -> np.ndarray:
    """A band of ``width`` pixels just outside (or just inside) the mask boundary."""
    m = mask.astype(bool)
    grown = box_mean(m.astype(np.float64), 2 * int(width) + 1) > 1e-9
    if outside:
        return grown & ~m
    shrunk = box_mean((~m).astype(np.float64), 2 * int(width) + 1) > 1e-9
    return m & shrunk


def seam_contrast(image: np.ndarray, mask: np.ndarray, width: int = 3) -> dict:
    """Mean absolute step across the mask boundary, with a control band for scale.

    The boundary step alone says nothing: a real panel edge is a step too. So the same
    measurement is taken across a boundary of identical shape translated into untouched
    background, and the two are reported together. A seam that reads far above the control
    is a compositing artefact; one at the control's level is scene structure.
    """
    grey = image.astype(np.float64)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    inner = ring(mask, width, outside=False)
    outer = ring(mask, width, outside=True)
    if not inner.any() or not outer.any():
        return {"usable": False,
                "reason": "mask too small or too large for a boundary band"}
    step = abs(float(grey[inner].mean() - grey[outer].mean()))
    # Control: the same bands shifted far enough to leave the composited region.
    h, w = grey.shape
    ys, xs = np.nonzero(mask)
    shift = int(min(max(xs.max() - xs.min(), 1) * 2, w - 1 - xs.max())) if xs.size else 0
    ctrl = None
    if shift > 2:
        ci = np.roll(inner, shift, axis=1)
        co = np.roll(outer, shift, axis=1)
        if ci.any() and co.any():
            ctrl = abs(float(grey[ci].mean() - grey[co].mean()))
    return {
        "usable": True,
        "band_width_px": int(width),
        "boundary_step": round(step, 3),
        "control_step": round(ctrl, 3) if ctrl is not None else None,
        "control_shift_px": shift if ctrl is not None else None,
        "_note": ("Both numbers are mean grey levels out of 255. The control band is the "
                  "same boundary shape shifted sideways into untouched scene, so it gives "
                  "the step an ordinary scene edge produces on this footage."),
    }


def region_stability(frames: np.ndarray, mask: np.ndarray) -> dict:
    """How much the masked region moves over a clip, relative to its own median.

    A fixed donor patch can only replace a region that holds still. This returns the mean
    absolute deviation from the temporal median inside the mask and, for scale, the same
    quantity over the whole frame.
    """
    arr = frames.astype(np.float64)
    grey = arr.mean(axis=3) if arr.ndim == 4 else arr
    med = np.median(grey, axis=0)
    dev = np.abs(grey - med)
    m = mask.astype(bool)
    return {
        "frames": int(len(grey)),
        "mask_px": int(m.sum()),
        "mad_in_mask": round(float(dev[:, m].mean()), 4) if m.any() else None,
        "mad_whole_frame": round(float(dev.mean()), 4),
        "max_mad_in_mask": round(float(dev[:, m].max()), 4) if m.any() else None,
    }
