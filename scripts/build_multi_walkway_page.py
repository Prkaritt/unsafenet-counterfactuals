#!/usr/bin/env python
"""Build the HTML review page for the multi-source base-versus-LoRA comparison.

One section per source: the three clips playable side by side, the numbers beside
them, and a plain statement of what each number does and does not establish. The
page carries no signature field; the review stays unsigned until a human completes
REVIEW_TEMPLATE.md.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def num(v, d=1):
    if v is None:
        return "&mdash;"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return html.escape(str(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/multi_walkway/review")
    args = ap.parse_args()

    mw = REPO / "outputs" / "multi_walkway"
    plan = json.loads((mw / "generation_plan.json").read_text())
    ev = json.loads((mw / "evaluation.json").read_text())["clips"]
    man_p = mw / "manifest_multi_walkway.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}

    out = REPO / args.out
    (out / "media").mkdir(parents=True, exist_ok=True)

    def copy(src: Path) -> str | None:
        if not src.exists():
            return None
        shutil.copy2(src, out / "media" / src.name)
        return f"media/{src.name}"

    secs = []
    for s in plan["sources"]:
        cid = s["clip_id"]
        e = ev.get(cid, {})
        media = {
            "source": copy(REPO / s["preprocessed"]),
            "base": copy(mw / "videos" / f"{cid}_base.mp4"),
            "lora": copy(mw / "videos" / f"{cid}_lora.mp4"),
        }
        overlay = copy(mw / "overlays" / f"{cid}_footpath.png")
        rows = []
        for arm in ("source", "base", "lora"):
            a = e.get(arm, {})
            lp = a.get("lane_preservation", {})
            rows.append(
                f"<tr><th>{arm}</th>"
                f"<td>{num(a.get('dist_to_lane_px_median'))}</td>"
                f"<td>{num(a.get('dist_to_lane_px_min'))}</td>"
                f"<td>{num(a.get('delta_dist_vs_source_px'))}</td>"
                f"<td>{num(a.get('frames_on_lane'))} / {num(a.get('frames'))}</td>"
                f"<td>{num(a.get('detector_misses'))}</td>"
                f"<td>{num(lp.get('iou_with_source_lane'), 3)}</td>"
                f"<td>{num(a.get('psnr_vs_source_db'), 3)}</td>"
                f"<td>{num(a.get('mean_abs_diff_vs_source'), 3)}</td></tr>")
        vids = "".join(
            f'<figure><figcaption>{k}</figcaption>'
            f'<video src="{v}" controls loop muted playsinline></video></figure>'
            if v else f"<figure><figcaption>{k} (missing)</figcaption></figure>"
            for k, v in media.items())
        secs.append(f"""
<section>
  <h2>{html.escape(cid)}</h2>
  <p class="role">{html.escape(s.get('role',''))}</p>
  <p class="why"><strong>Why selected:</strong> {html.escape(s.get('selection_reason',''))}</p>
  <div class="videos">{vids}</div>
  {f'<p><img src="{overlay}" alt="foot path overlay"></p>' if overlay else ''}
  <table>
    <thead><tr><th>clip</th><th>foot&rarr;lane median px</th><th>min px</th>
      <th>&Delta; vs source</th><th>frames on lane</th><th>detector misses</th>
      <th>lane IoU</th><th>PSNR dB</th><th>MAD</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <details><summary>prompt</summary><pre>{html.escape(s['prompt'])}</pre></details>
</section>""")

    st = plan["settings"]
    ad = man.get("adapter", {})
    page = f"""<!doctype html>
<meta charset="utf-8"><title>Multi-walkway base vs LoRA review</title>
<style>
 body{{font:15px/1.55 system-ui,sans-serif;margin:0 auto;max-width:1150px;padding:24px;color:#111}}
 h1{{margin-bottom:4px}} section{{border-top:2px solid #ddd;margin-top:32px;padding-top:16px}}
 .videos{{display:flex;gap:12px;flex-wrap:wrap}} figure{{margin:0}}
 video{{width:350px;background:#000;border-radius:4px}}
 figcaption{{font-weight:600;margin-bottom:4px}}
 table{{border-collapse:collapse;margin-top:14px;font-size:13px;width:100%}}
 th,td{{border:1px solid #ccc;padding:5px 8px;text-align:right}}
 thead th{{background:#f4f4f4;text-align:center}} tbody th{{text-align:left}}
 .warn{{background:#fff6e5;border-left:4px solid #e0a106;padding:12px 16px;margin:16px 0}}
 .role{{font-weight:600}} .why{{color:#444}} pre{{white-space:pre-wrap}}
 img{{max-width:100%}}
</style>
<h1>Walkway counterfactuals: base CogVideoX-2b vs the trained LoRA adapter</h1>
<p>Three walkway-violation sources, each edited twice. Within a pair the only difference is
the adapter.</p>

<div class="warn">
<p><strong>These are candidate counterfactuals, not verified safety.</strong> A worker moving
onto a painted lane is not proof of safe walking, and no classifier prediction is used as
evidence here. Classifier-gradient guidance was disabled in every run.</p>
<p><strong>What the numbers mean.</strong> <em>foot&rarr;lane</em> is the distance from the
worker's feet to the nearest pixel of <em>that source's own</em> lane, annotated once from
the preprocessed source and applied unchanged to all three clips, so a model that repaints
the floor cannot satisfy it. A negative &Delta; means the feet ended closer to the lane than
in the source. <em>lane IoU</em> compares each clip's own green paint with the source's: well
below 1 means the marking itself moved, widened or vanished. <em>PSNR</em> and <em>MAD</em>
are whole-frame statistics dominated by the static background and nearly blind to a worker
occupying a small part of the frame &mdash; do not read them as evidence that the worker was
preserved.</p>
<p><strong>Only two camera views exist</strong> among eligible sources, not three. 0_tr16 and
0_tr48 share a view (background correlation r = 0.929) from different recordings; 0_tr125 is
the second view. The third source supplies a different walking direction instead.</p>
<p>Three clips cannot support a claim about generalisation.</p>
</div>

<h2>Settings, identical within every pair</h2>
<table><tbody>
<tr><th>seed</th><td>{st['seed']}</td></tr>
<tr><th>strength</th><td>{st['strength']}</td></tr>
<tr><th>text guidance</th><td>{st['guidance_scale']}</td></tr>
<tr><th>requested steps</th><td>{st['num_inference_steps']}</td></tr>
<tr><th>resolution, frames</th><td>{st['width']}x{st['height']}, {st['num_frames']} at {st['fps']} fps</td></tr>
<tr><th>scheduler</th><td>{html.escape(str(man.get('scheduler','—')))}</td></tr>
<tr><th>checkpoint</th><td>{html.escape(str(man.get('checkpoint','THUDM/CogVideoX-2b')))} @ {html.escape(str(man.get('revision','—'))[:12])}</td></tr>
<tr><th>adapter sha256</th><td style="font-family:monospace;font-size:11px">{html.escape(str(ad.get('sha256','—')))}</td></tr>
<tr><th>adapter scale</th><td>{num(ad.get('scale_applied'),2)} (rank {ad.get('rank','—')}, alpha {ad.get('lora_alpha','—')})</td></tr>
<tr><th>LoRA tensors, base arm / LoRA arm</th><td>{ad.get('lora_param_tensors_before','—')} / {ad.get('lora_param_tensors_after','—')}</td></tr>
<tr><th>classifier guidance</th><td>disabled in every run</td></tr>
</tbody></table>
{''.join(secs)}
<section><h2>Review status</h2>
<p>Unsigned. Complete <code>REVIEW_TEMPLATE.md</code> against each source, judging the four
criteria separately, and reject explicitly any clip whose lane was repainted, widened or
extended under a worker who did not move.</p></section>
"""
    (out / "index.html").write_text(page)
    print(f"wrote {out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
