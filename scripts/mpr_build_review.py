#!/usr/bin/env python
"""Stage the downloadable review package and build its self-contained HTML page.

Everything in the package uses relative paths, so the page works from a local folder after
an rsync with no server and no network. Each clip gets three players side by side -- source,
rough composite, refined -- plus its annotation overlays and its measured numbers.

The clips show identifiable workers who consented to the original study and not to
redistribution. The package is for local review. Nothing here is uploaded anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "manual_path"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verdict_block(cid: str, ev: dict, comp: dict, ref: dict | None) -> str:
    rows = []

    def arm_row(name: str, m: dict) -> str:
        ai = m.get("after_intervention", {})
        return (
            f"<tr><td>{html.escape(name)}</td>"
            f"<td class='n'>{m.get('dist_to_lane_px_median')}</td>"
            f"<td class='n'>{ai.get('dist_to_lane_px_median')}</td>"
            f"<td class='n'>{m.get('frames_on_lane')}/{m.get('frames')}</td>"
            f"<td class='n'>{ai.get('frames_on_lane')}/{ai.get('frames')}</td>"
            f"<td class='n'>{m.get('detector_misses')}</td>"
            f"<td class='n'>{m.get('lane_preservation', {}).get('iou_with_source_lane')}</td>"
            f"<td class='n'>{m.get('psnr_vs_source_db', '—')}</td></tr>")

    for name in ("source", "composite", "refined"):
        if name in ev:
            rows.append(arm_row(name, ev[name]))

    return (
        "<table class='data'><thead><tr>"
        "<th>arm</th><th class='n'>foot-to-lane median, px</th>"
        "<th class='n'>after intervention</th>"
        "<th class='n'>frames on lane</th><th class='n'>after intervention</th>"
        "<th class='n'>detector misses</th><th class='n'>lane IoU</th>"
        "<th class='n'>PSNR vs source, dB</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def build_html(pkg: Path, clips: list[str], ev: dict, findings_md: str) -> str:
    sections = []
    for cid in clips:
        d = json.loads((pkg / "03_manifests" / f"{cid}_annotation.json").read_text())
        comp = json.loads((pkg / "03_manifests" / f"{cid}_composite.json").read_text())
        ref_p = pkg / "03_manifests" / f"{cid}_refine.json"
        ref = json.loads(ref_p.read_text()) if ref_p.exists() else None
        e = ev["clips"].get(cid, {})

        players = []
        for label, fname, note in (
            ("Source", f"{cid}_source.mp4",
             "the interval exactly as preprocessed, before any intervention"),
            ("Rough composite", f"{cid}_composite.mp4",
             "worker segmented, background recovered, silhouette moved along the "
             "agent-specified path. No diffusion."),
            ("CogVideoX refined", f"{cid}_refined.mp4",
             "frozen CogVideoX-2b video-to-video over the composite, no LoRA, no "
             "classifier gradients"),
        ):
            p = pkg / "01_videos" / fname
            if not p.exists():
                players.append(
                    f"<div class='player missing'><h4>{label}</h4>"
                    f"<p class='miss'>not produced — see findings</p></div>")
                continue
            players.append(
                f"<div class='player'><h4>{html.escape(label)}</h4>"
                f"<video src='01_videos/{fname}' controls loop muted playsinline "
                f"preload='metadata' data-group='{cid}'></video>"
                f"<p class='cap'>{html.escape(note)}</p></div>")

        ov = []
        for fname, cap in (
            (f"{cid}_01_green_components.png",
             "Green components, numbered and coloured. Chosen component(s): "
             f"{d['lane']['components_chosen']}. Green paint is a candidate for the lane, "
             "not proof of one."),
            (f"{cid}_03_lane_path.png",
             "Lane polygon in green, obstacle polygons in red, the worker's real foot "
             "track in white, and the agent-specified target path in orange with its "
             "keyframes marked."),
            (f"{cid}_04_background_estimate.png",
             "Temporal-median background with every detected person withheld. Machinery "
             "that moves is flattened to whichever state held for most of the clip."),
            (f"{cid}_05_worker_masks.png",
             "Every sixth frame with the propagated worker mask tinted and the target foot "
             "point marked. Inspect limbs, identity continuity and occlusions here."),
        ):
            if (pkg / "02_overlays" / fname).exists():
                ov.append(f"<figure><img src='02_overlays/{fname}' alt='{html.escape(cap)}'>"
                          f"<figcaption>{html.escape(cap)}</figcaption></figure>")

        ch = d["path_checks"]
        settings = ""
        if ref:
            settings = (
                f"<p class='settings'>seed {ref['seed']} &middot; strength {ref['strength']} "
                f"&middot; {ref['denoising_steps_executed']} of {ref['num_inference_steps']} "
                f"steps executed &middot; guidance {ref['guidance_scale']} &middot; "
                f"{html.escape(ref['scheduler'])} &middot; {html.escape(ref['dtype'])} "
                f"&middot; LoRA tensors present: {ref['lora_param_tensors_present']}</p>")

        sections.append(f"""
<section id="{html.escape(cid)}">
  <h2>{html.escape(cid)}</h2>
  <p class="lede">{html.escape(d['intervention']['rationale'])}</p>
  <p class="agent">Agent-specified intervention: this path was proposed from image evidence.
  No model discovered it and it is not present in the dataset.</p>
  {settings}
  <div class="row">
    <button class="playrow" type="button" data-group="{html.escape(cid)}">Play all three from the start</button>
    <span class="meta">intervention starts at frame {d['intervention']['first_frame']} &middot;
    {ch['frames_relocated']} frames relocated &middot;
    max lateral step {ch['max_lateral_step_px']} px (limit {ch['speed_limit_px']}) &middot;
    teleports {ch['teleports']} &middot; crosses machinery {ch['crosses_machinery']}</span>
  </div>
  <div class="players">{''.join(players)}</div>
  {verdict_block(cid, e, comp, ref)}
  <details><summary>Annotation overlays</summary><div class="overlays">{''.join(ov)}</div></details>
  <details><summary>Method limits that apply to this clip</summary><ul>
    {''.join(f'<li>{html.escape(x)}</li>' for x in comp['composite']['_limits'])}
    {''.join(f'<li>{html.escape(x)}</li>' for x in d['geometry_uncertainty'])}
  </ul></details>
</section>""")

    findings_html = "".join(
        f"<p>{html.escape(line)}</p>" for line in findings_md.splitlines()
        if line.strip() and not line.startswith("#"))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Manual path relocation review</title>
<style>
:root {{ --bg:#f4f3ef; --fg:#1d201c; --soft:#5a5f56; --rule:#cfccc2; --surface:#fbfaf7;
         --lane:#2f7d4f; --fail:#a8382a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#151714; --fg:#e8e6de; --soft:#a0a497;
         --rule:#3a3e34; --surface:#1d201c; --lane:#6cc08c; --fail:#e08273; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 Georgia, serif; }}
.wrap {{ max-width:1200px; margin:0 auto; padding:0 16px; }}
header {{ border-bottom:2px solid var(--fg); padding:32px 0 16px; }}
h1 {{ font-family:system-ui, sans-serif; font-size:2rem; margin:0 0 8px; }}
h2 {{ font-family:system-ui, sans-serif; font-size:1.4rem; margin:0 0 6px; }}
h4 {{ font-family:ui-monospace, monospace; font-size:.75rem; letter-spacing:.08em;
      text-transform:uppercase; color:var(--soft); margin:0 0 6px; }}
section {{ padding:28px 0; border-top:1px solid var(--rule); }}
.lede {{ max-width:70ch; }}
.agent {{ max-width:70ch; color:var(--fail); font-size:.9rem; }}
.settings, .meta {{ font-family:ui-monospace, monospace; font-size:.75rem; color:var(--soft); }}
.row {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin:10px 0; }}
button.playrow {{ font-family:ui-monospace, monospace; font-size:.75rem; text-transform:uppercase;
  letter-spacing:.08em; padding:6px 12px; background:var(--surface); color:var(--fg);
  border:1px solid var(--soft); cursor:pointer; }}
.players {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px,1fr)); gap:14px; }}
.player {{ border:1px solid var(--rule); background:var(--surface); padding:10px; }}
.player.missing {{ border-color:var(--fail); }}
.miss {{ color:var(--fail); font-size:.85rem; }}
video {{ width:100%; height:auto; display:block; background:#000; }}
.cap {{ font-size:.8rem; color:var(--soft); margin:6px 0 0; }}
table.data {{ border-collapse:collapse; width:100%; margin:16px 0; font-family:ui-monospace,
  monospace; font-size:.78rem; }}
table.data th, table.data td {{ border-bottom:1px solid var(--rule); padding:5px 8px;
  text-align:left; }}
table.data th {{ color:var(--soft); font-size:.68rem; text-transform:uppercase; }}
td.n, th.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
details {{ margin:12px 0; }}
summary {{ cursor:pointer; font-family:system-ui, sans-serif; font-size:.9rem; }}
.overlays figure {{ margin:12px 0; }}
.overlays img {{ max-width:100%; border:1px solid var(--rule); display:block; }}
figcaption {{ font-size:.8rem; color:var(--soft); margin-top:6px; max-width:70ch; }}
footer {{ border-top:2px solid var(--fg); padding:24px 0 48px; font-size:.85rem;
  color:var(--soft); }}
ul {{ max-width:75ch; }}
</style></head><body><div class="wrap">
<header>
  <h1>Manual path relocation review</h1>
  <p class="lede">Three source clips. In each, the worker is explicitly relocated onto the
  marked pedestrian lane by segmentation and compositing, and a frozen CogVideoX-2b is then
  asked only to refine the result. Play the three videos in each section against each other:
  source, rough composite, refined.</p>
  <p class="agent">Every path here is an agent-specified intervention, not discovered safe
  behaviour. Nothing on this page is a safety claim, and no clip is a verified
  counterfactual.</p>
</header>
<section id="findings"><h2>Findings</h2>{findings_html}</section>
{''.join(sections)}
<footer>
  <p>Clips show identifiable workers who consented to the original study and not to
  redistribution. Keep this package local.</p>
  <p>Unsafe-Net dataset, CC BY 4.0, doi:10.17632/xjmtb22pff.1. CogVideoX-2b, Apache 2.0.
  SAM 2.1, Apache 2.0. Verify the package with
  <code>shasum -a 256 -c SHA256SUMS.txt</code>.</p>
</footer>
</div>
<script>
document.addEventListener('click', function (e) {{
  var b = e.target.closest('button.playrow');
  if (!b) return;
  var g = b.getAttribute('data-group');
  document.querySelectorAll("video[data-group='" + g + "']").forEach(function (v) {{
    try {{ v.currentTime = 0; v.play(); }} catch (err) {{}}
  }});
}});
</script>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/ocean/projects/cis260288p/puprety/transfer/manual-path-review")
    ap.add_argument("--mac-dest",
                    default=str(Path.home() / "workspace/unsafenet-counterfactuals/"
                                              "unsafenet_cf_outputs/manual-path-review"))
    ap.add_argument("--ssh-user", default=os.environ.get("USER", "<user>"),
                    help="Bridges-2 login name used in the printed rsync command")
    ap.add_argument("--sources", default="configs/manual_path_sources.json")
    args = ap.parse_args()

    clips = [s["clip_id"] for s in json.loads((REPO / args.sources).read_text())["sources"]]
    pkg = Path(args.dest)
    if pkg.exists():
        shutil.rmtree(pkg)
    for sub in ("01_videos", "02_overlays", "03_manifests", "04_logs"):
        (pkg / sub).mkdir(parents=True)

    copied: list[Path] = []

    def put(src: Path, rel: str) -> None:
        if not src.exists():
            print(f"  missing, not copied: {src}")
            return
        dst = pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)

    for cid in clips:
        d = OUT / cid
        for arm in ("source", "composite", "refined"):
            put(d / "videos" / f"{cid}_{arm}.mp4", f"01_videos/{cid}_{arm}.mp4")
        for png in sorted((d / "overlays").glob("*.png")):
            put(png, f"02_overlays/{cid}_{png.name}")
        for name in ("annotation", "composite", "refine", "survey"):
            put(d / f"{name}.json", f"03_manifests/{cid}_{name}.json")

    put(OUT / "evaluation.json", "03_manifests/evaluation.json")
    put(OUT / "refine_manifest.json", "03_manifests/refine_manifest.json")
    # The two files behind the lane-IoU correction. Without them the write-up asserts that
    # the collapse is a colour shift rather than a repainting and the reader cannot check it.
    put(OUT / "scale_drift.json", "03_manifests/scale_drift.json")
    put(OUT / "lane_colour.json", "03_manifests/lane_colour.json")
    put(REPO / "configs" / "manual_path_sources.json", "03_manifests/manual_path_sources.json")
    put(REPO / "configs" / "manual_path_plan.json", "03_manifests/manual_path_plan.json")
    put(REPO / "configs" / "manual_path_refine.json", "03_manifests/manual_path_refine.json")
    put(REPO / "DECISIONS.md", "DECISIONS.md")
    put(OUT / "FINDINGS.md", "FINDINGS.md")
    for log in sorted((REPO / "logs").glob("mpr*")):
        put(log, f"04_logs/{log.name}")

    ev_path = OUT / "evaluation.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {"clips": {}}
    findings_md = (OUT / "FINDINGS.md").read_text() if (OUT / "FINDINGS.md").exists() else ""
    index = pkg / "index.html"
    index.write_text(build_html(pkg, clips, ev, findings_md))
    copied.append(index)

    lines = [f"{sha256(p)}  {p.relative_to(pkg)}" for p in sorted(copied, key=str)]
    (pkg / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")

    rsync = (f"rsync -avP --partial "
             f"{args.ssh_user}@data.bridges2.psc.edu:{pkg}/ {args.mac_dest}/")
    (pkg / "HOW_TO_DOWNLOAD.txt").write_text(
        "Run on your own machine, not on Bridges-2:\n\n"
        f"    mkdir -p {args.mac_dest}\n    {rsync}\n\n"
        "Verify:\n\n"
        f"    cd {args.mac_dest} && shasum -a 256 -c SHA256SUMS.txt\n\n"
        "Open:\n\n"
        f"    open {args.mac_dest}/index.html\n\n"
        "The page uses relative paths and needs no server and no network.\n"
        "The clips show identifiable workers who consented to the original study and not\n"
        "to redistribution. Keep this package local.\n")

    total = sum(p.stat().st_size for p in copied)
    print(f"\nstaged {len(copied)} files, {total / 2**20:.1f} MiB at {pkg}")
    print(f"\nDownload:\n    mkdir -p {args.mac_dest}\n    {rsync}")
    print(f"\nOpen:\n    open {args.mac_dest}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
