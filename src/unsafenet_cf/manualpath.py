"""Geometry for the agent-specified worker-relocation experiment.

Everything here works in the image plane of the *preprocessed* clip, 720x480. That is a
hard limit on what any of it can mean. A path drawn in pixels is a path across a
photograph: it carries no camera calibration, no ground plane, and no metric scale, so it
cannot establish a real-world distance, a clearance, or a safe separation from a machine.
What it can do is place a silhouette inside a region that was annotated from the same
footage, and that is the whole claim.

Dependencies are deliberately numpy-only. ``scipy`` is not in the pinned Bridges-2
environment, so connected components, the convex hull and the point-in-polygon test are
written out rather than imported.
"""

from __future__ import annotations

from collections import deque

import numpy as np


# --------------------------------------------------------------------------- regions

def connected_components(mask: np.ndarray, min_area: int = 40) -> list[dict]:
    """Label 8-connected True regions, largest first.

    Iterates only over set pixels, so cost follows the mask's area and not the frame's.
    Components smaller than ``min_area`` are dropped: on this footage they are paint
    speckle and compression noise, never a lane marking.
    """
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    out: list[dict] = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        q = deque([(sy, sx)])
        seen[sy, sx] = True
        pix: list[tuple[int, int]] = []
        while q:
            y, x = q.popleft()
            pix.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        if len(pix) < min_area:
            continue
        arr = np.array(pix)
        cy, cx = arr[:, 0], arr[:, 1]
        out.append({
            "area": int(len(pix)),
            "bbox": [int(cx.min()), int(cy.min()), int(cx.max()), int(cy.max())],
            "centroid_xy": [round(float(cx.mean()), 1), round(float(cy.mean()), 1)],
            "y_span": int(cy.max() - cy.min()) + 1,
            "x_span": int(cx.max() - cx.min()) + 1,
            "pixels": arr,
        })
    out.sort(key=lambda c: -c["area"])
    return out


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain hull of (N, 2) xy points, counter-clockwise."""
    p = np.unique(points.astype(np.float64), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    if len(p) <= 2:
        return p

    def half(seq):
        out: list[np.ndarray] = []
        for q in seq:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    lower = half(p)
    upper = half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Ray-crossing test. Points exactly on an edge may fall either way."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xx = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xx:
                inside = not inside
    return inside


def polygon_mask(poly: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a polygon by scanline. Shape is (H, W)."""
    h, w = shape
    m = np.zeros((h, w), bool)
    if len(poly) < 3:
        return m
    for y in range(h):
        xs = []
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        xs.sort()
        for a, b in zip(xs[0::2], xs[1::2]):
            x0 = max(0, int(np.ceil(a)))
            x1i = min(w - 1, int(np.floor(b)))
            if x1i >= x0:
                m[y, x0:x1i + 1] = True
    return m


# --------------------------------------------------------------- perspective and paths

def fit_height_vs_y(foot_y: list[float], height: list[float]) -> dict:
    """Least-squares height = a * foot_y + b, with the fit quality kept.

    This is the only perspective evidence available without camera calibration: as the
    worker walks, the detector's box height changes with the foot row, and that pairing
    is measured from the footage itself. It is a one-dimensional stand-in for a ground
    plane. It holds only over the foot rows the worker actually occupied, and the
    returned ``y_range`` says which those were. Extrapolating past them is guessing.
    """
    y = np.asarray(foot_y, float)
    hh = np.asarray(height, float)
    if len(y) < 3 or float(y.max() - y.min()) < 1.0:
        return {"usable": False, "reason": "fewer than 3 points, or no spread in foot row",
                "n": int(len(y))}
    a, b = np.polyfit(y, hh, 1)
    pred = a * y + b
    ss_res = float(((hh - pred) ** 2).sum())
    ss_tot = float(((hh - hh.mean()) ** 2).sum())
    return {
        "usable": True,
        "slope_px_height_per_px_row": round(float(a), 4),
        "intercept_px": round(float(b), 2),
        "r_squared": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
        "n": int(len(y)),
        "y_range": [round(float(y.min()), 1), round(float(y.max()), 1)],
        "height_range": [round(float(hh.min()), 1), round(float(hh.max()), 1)],
        "_meaning": ("Empirical image-plane scale. Not a calibrated ground plane, and valid "
                     "only across y_range."),
    }


def scale_at(fit: dict, foot_y: float, reference_height: float) -> float:
    """Height multiplier for a silhouette moved to foot row ``foot_y``.

    Returns 1.0 when the fit is unusable or when the target row is outside the range the
    fit was measured over, so an unsupported extrapolation never silently rescales a
    person. The caller records that this happened.
    """
    if not fit.get("usable") or reference_height <= 0:
        return 1.0
    lo, hi = fit["y_range"]
    if not (lo - 1e-6 <= foot_y <= hi + 1e-6):
        return 1.0
    want = fit["slope_px_height_per_px_row"] * foot_y + fit["intercept_px"]
    if want <= 1.0:
        return 1.0
    return float(want / reference_height)


def interpolate_path(keyframes: list[dict], n_frames: int) -> list[dict]:
    """Piecewise-linear foot position per frame from (frame, x, y) keyframes.

    Frames before the first keyframe keep ``None``, which the compositor reads as "leave
    this frame exactly as the source shot it". That is what preserves the original
    sequence before the intervention instead of rewriting the whole clip.
    """
    kf = sorted(keyframes, key=lambda k: k["frame"])
    if not kf:
        return [None] * n_frames
    out: list[dict | None] = [None] * n_frames
    for i in range(len(kf) - 1):
        a, b = kf[i], kf[i + 1]
        span = b["frame"] - a["frame"]
        for f in range(a["frame"], min(b["frame"], n_frames - 1) + 1):
            t = 0.0 if span == 0 else (f - a["frame"]) / span
            out[f] = {"frame": f,
                      "x": a["x"] + t * (b["x"] - a["x"]),
                      "y": a["y"] + t * (b["y"] - a["y"])}
    last = kf[-1]
    for f in range(last["frame"], n_frames):
        out[f] = {"frame": f, "x": float(last["x"]), "y": float(last["y"])}
    return out


def path_checks(path, source_feet, lane_poly, obstacle_polys, speed_limit_px) -> dict:
    """Everything that could make an agent-specified path implausible, measured.

    Reported, never silently corrected: a path that teleports or crosses machinery is a
    finding about the proposed intervention, not something to quietly smooth away.
    """
    # Two step measures, because they answer different questions. The lateral step is the
    # intervention's own doing and is what the speed limit governs. The total step also
    # contains the worker's real vertical motion, which the path copies frame for frame
    # under follow_source, so a total above the limit can be the footage's own jitter
    # rather than the intervention moving anyone impossibly fast. Both are reported.
    steps, lat_steps, over = [], [], []
    prev = None
    for p in path:
        if p is None:
            prev = None
            continue
        if prev is not None:
            d = float(np.hypot(p["x"] - prev["x"], p["y"] - prev["y"]))
            dx = abs(float(p["x"] - prev["x"]))
            steps.append(d)
            lat_steps.append(dx)
            if dx > speed_limit_px:
                over.append({"frame": p["frame"], "lateral_step_px": round(dx, 1),
                             "total_step_px": round(d, 1)})
        prev = p

    in_obstacle = []
    for p in path:
        if p is None:
            continue
        for name, poly in obstacle_polys.items():
            if point_in_polygon(p["x"], p["y"], poly):
                in_obstacle.append({"frame": p["frame"], "obstacle": name})

    first = next((p for p in path if p is not None), None)
    start_gap = None
    if first is not None and source_feet[first["frame"]] is not None:
        sx, sy = source_feet[first["frame"]]
        start_gap = round(float(np.hypot(first["x"] - sx, first["y"] - sy)), 2)

    ends_in_lane = None
    last = next((p for p in reversed(path) if p is not None), None)
    if last is not None and lane_poly is not None and len(lane_poly) >= 3:
        ends_in_lane = point_in_polygon(last["x"], last["y"], lane_poly)

    frames_in_lane = sum(
        1 for p in path
        if p is not None and lane_poly is not None and len(lane_poly) >= 3
        and point_in_polygon(p["x"], p["y"], lane_poly))

    return {
        "max_lateral_step_px": round(max(lat_steps), 2) if lat_steps else None,
        "median_lateral_step_px": round(float(np.median(lat_steps)), 2) if lat_steps else None,
        "max_total_step_px": round(max(steps), 2) if steps else None,
        "median_total_step_px": round(float(np.median(steps)), 2) if steps else None,
        "_total_includes": "the worker's own vertical motion, copied from the source track",
        "speed_limit_px": round(float(speed_limit_px), 2),
        "steps_over_limit": over,
        "teleports": bool(over),
        "frames_inside_obstacle": in_obstacle,
        "crosses_machinery": bool(in_obstacle),
        "start_offset_from_source_foot_px": start_gap,
        "ends_inside_lane_polygon": ends_in_lane,
        "frames_with_foot_inside_lane_polygon": frames_in_lane,
        "frames_relocated": sum(1 for p in path if p is not None),
    }


# ------------------------------------------------------------------------- compositing

def temporal_background(frames: np.ndarray, exclude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median over frames where ``exclude`` is False.

    ``exclude`` is (T, H, W) and marks pixels to withhold, which here means every person
    the segmenter found in that frame. A pixel withheld in *every* frame has no evidence
    behind it; those pixels are returned in the second array so the caller reports them
    instead of pretending the background is known there.

    Machinery is a separate problem this cannot solve. A press that moves through the
    sequence is sometimes in and sometimes out of a given pixel, and a median over the
    whole clip returns whichever state held for more than half the frames. That makes the
    estimate a plausible still, not a true empty room, and any press caught mid-stroke in
    the source may be flattened to its resting position.
    """
    t, h, w, _ = frames.shape
    bg = np.zeros((h, w, 3), np.uint8)
    unknown = np.zeros((h, w), bool)
    keep = ~exclude
    counts = keep.sum(0)
    unknown |= counts == 0
    flat = frames.reshape(t, h * w, 3)
    kflat = keep.reshape(t, h * w)
    cflat = counts.reshape(h * w)
    bgflat = bg.reshape(h * w, 3)
    for idx in np.nonzero(cflat > 0)[0]:
        sel = flat[kflat[:, idx], idx, :]
        bgflat[idx] = np.median(sel, axis=0).astype(np.uint8)
    # Pixels with no evidence fall back to the whole-clip median, and are flagged.
    if unknown.any():
        med = np.median(flat, axis=0).astype(np.uint8).reshape(h, w, 3)
        bg[unknown] = med[unknown]
    return bg, unknown


def paste_silhouette(canvas: np.ndarray, patch: np.ndarray, alpha: np.ndarray,
                     foot_xy: tuple[float, float], scale: float = 1.0) -> dict:
    """Alpha-composite a silhouette so its foot point lands on ``foot_xy``.

    Nearest-neighbour resampling, chosen so that no interpolated colour is invented along
    the silhouette edge. The patch is the worker exactly as photographed: translating it
    cannot turn the body, change the gait, or move a cast shadow, and the shadow that was
    under the worker in the source stays wherever the background estimate put it.
    """
    ph, pw = alpha.shape
    if scale != 1.0:
        nh, nw = max(1, int(round(ph * scale))), max(1, int(round(pw * scale)))
        yi = np.clip((np.arange(nh) / scale).astype(int), 0, ph - 1)
        xi = np.clip((np.arange(nw) / scale).astype(int), 0, pw - 1)
        patch = patch[yi][:, xi]
        alpha = alpha[yi][:, xi]
        ph, pw = nh, nw

    fx, fy = foot_xy
    x0 = int(round(fx - pw / 2.0))
    y0 = int(round(fy - ph))
    h, w, _ = canvas.shape
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    cw = min(pw - sx0, w - dx0)
    ch = min(ph - sy0, h - dy0)
    if cw <= 0 or ch <= 0:
        return {"pasted": False, "reason": "silhouette falls entirely outside the frame"}
    a = alpha[sy0:sy0 + ch, sx0:sx0 + cw][..., None].astype(np.float32)
    src = patch[sy0:sy0 + ch, sx0:sx0 + cw].astype(np.float32)
    dst = canvas[dy0:dy0 + ch, dx0:dx0 + cw].astype(np.float32)
    canvas[dy0:dy0 + ch, dx0:dx0 + cw] = (a * src + (1.0 - a) * dst).astype(np.uint8)
    return {"pasted": True, "scale": round(float(scale), 4),
            "clipped": bool(cw != pw or ch != ph),
            "dest_bbox": [dx0, dy0, dx0 + cw, dy0 + ch]}
