#!/usr/bin/env python
"""Stage the panel-cover review package and build a self-contained local HTML page.

The page uses relative paths only, so the directory works unchanged after it is copied to
a laptop. Every video in it is playable, labelled with the arm it came from, and grouped by
case with a single play control per case so the arms can be compared frame for frame.

What goes in: the videos, the region overlays and contact sheets, the donor provenance, the
masks' polygons, the settings and manifests, the Slurm logs, DECISIONS.md, FINDINGS.md, the
verified definitions record, and SHA256SUMS.txt over everything. What stays out: model
weights and the dataset itself.

The clips show identifiable factory workers. They are staged for local review and are not
uploaded anywhere by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "outputs" / "panel_cover"

ARMS = [
    ("source", "Preprocessed source", "Class 2, Opened Panel Cover. 49 frames, 720x480, "
                                      "8 fps. The input to every other arm."),
    ("donor", "Donor clip, preprocessed", "Class 6, Closed Panel Cover, training split. "
                                          "The pixels the panel region is replaced with."),
    ("vae_source", "VAE round trip of the source", "Encoded and decoded by the CogVideoX "
                                                   "VAE with no diffusion. The sharpness "
                                                   "ceiling for every generated arm."),
    ("base_edit", "Prompt-only edit", "CogVideoX asked by prompt alone to close the panel. "
                                      "No composite, no mask. The control for what a "
                                      "prompt can do."),
    ("composite", "Closed-panel composite", "The panel region replaced with registered "
                                            "donor pixels. The cover is closed in every "
                                            "frame. No closing motion was generated."),
    ("vae_composite", "VAE round trip of the composite", "The composite through the same "
                                                         "round trip, so it can be "
                                                         "compared with the refined arm "
                                                         "on equal terms."),
    ("refined", "CogVideoX-refined composite", "A low-strength video-to-video pass over "
                                               "the composite. The whole generated frame "
                                               "is kept here, including anything the "
                                               "model changed outside the panel."),
    ("refined_restricted", "Refined, output-composited back",
     "The refined panel region placed back over the composite. This is output compositing "
     "applied after generation, NOT mask-conditioned generation: the model was never "
     "given a mask."),
    ("unguided_control", "Unguided control",
     "The same denoising loop at the same seed and settings as the guided run, with the "
     "classifier term switched off. It exists so that any difference in the guided run "
     "can be read against a run that differs in nothing else."),
    ("guided", "Classifier-guided",
     "The same loop with the classifier gradient added. A changed prediction here is a "
     "statement about the classifier, not about the panel, and is read against the "
     "unguided control and the same panel-state proxy as every other arm."),
]


def log(m: str) -> None:
    print(m, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def esc(x) -> str:
    return html.escape(str(x))


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "&mdash;"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return esc(value)


def build_page(cases: list[dict], meta: dict) -> str:
    rows = []
    for case in cases:
        cid = case["case_id"]
        arms_html = []
        for key, title, blurb in ARMS:
            rel = f"videos/{cid}/{cid}_{key}.mp4"
            if rel not in case["files"]:
                continue
            arms_html.append(f"""
      <figure class="arm">
        <figcaption><b>{esc(title)}</b><span>{esc(blurb)}</span></figcaption>
        <video class="clip" data-case="{esc(cid)}" src="{esc(rel)}"
               controls muted loop preload="metadata"></video>
      </figure>""")

        ev = case.get("evaluation", {}).get("arms", {})
        measure_rows = []
        for key, title, _ in ARMS:
            if key not in ev:
                continue
            e = ev[key]
            proxy = e.get("panel_state_proxy", {})
            pres = e.get("preservation", {})
            seam = e.get("seam", {})
            measure_rows.append(
                "<tr><td>" + esc(title) + "</td>"
                f"<td>{fmt(proxy.get('ncc_vs_open_reference'), 4)}</td>"
                f"<td>{fmt(proxy.get('ncc_vs_closed_reference'), 4)}</td>"
                f"<td>{fmt(e.get('gradient_energy_in_region'))}</td>"
                f"<td>{fmt(pres.get('psnr_outside_region_db'), 2)}</td>"
                f"<td>{fmt(seam.get('boundary_step'), 2)} / "
                f"{fmt(seam.get('control_step'), 2)}</td>"
                f"<td>{fmt(e.get('region_stability', {}).get('mad_in_mask'), 3)}</td>"
                "</tr>")

        cls = case.get("evaluation", {}).get("classifier", {})
        cls_rows = []
        for name, s in (cls.get("arms") or {}).items():
            cls_rows.append(
                f"<tr><td>{esc(name)}</td>"
                f"<td>{fmt(s.get('prob_closed_panel_cover'), 4)}</td>"
                f"<td>{s.get('predicted_binary_index')}</td></tr>")

        comp = case.get("composite", {})
        reg = comp.get("registration", {})
        checks = comp.get("checks", {})
        don = comp.get("donor_provenance", {})
        overlays = [f for f in case["files"] if f.startswith(f"overlays/{cid}/")]

        rows.append(f"""
  <section class="case" id="{esc(cid)}">
    <h2>{esc(cid)}</h2>
    <p class="pair">Source <code>{esc(case.get('source_clip'))}</code> (class 2, Opened
      Panel Cover) &rarr; donor <code>{esc(case.get('donor_clip'))}</code> (class 6,
      Closed Panel Cover, training split).</p>
    <div class="controls">
      <button data-play="{esc(cid)}">Play all arms together</button>
      <button data-pause="{esc(cid)}">Pause</button>
      <button data-reset="{esc(cid)}">Back to start</button>
    </div>
    <div class="arms">{''.join(arms_html)}</div>

    <h3>Registration and compositing checks</h3>
    <table class="kv">
      <tr><td>Transform</td><td>{esc(reg.get('transform'))}, fitted from
        {esc(reg.get('landmarks_used'))} landmark patches</td></tr>
      <tr><td>Reprojection residual</td><td>rms
        {fmt((reg.get('fit') or {}).get('residual_px', {}).get('rms'), 4)} px, max
        {fmt((reg.get('fit') or {}).get('residual_px', {}).get('max'), 4)} px, dof
        {esc((reg.get('fit') or {}).get('degrees_of_freedom'))}</td></tr>
      <tr><td>Donor coverage of the region</td>
        <td>{fmt(reg.get('donor_coverage_of_region'), 5)}</td></tr>
      <tr><td>Pixels changed outside the blended region</td><td>{esc(
        (checks.get('unchanged_outside_blended_region') or {})
        .get('pixels_changed_outside_mask'))}</td></tr>
      <tr><td>Region size</td><td>{esc((comp.get('region') or {}).get('mask_px'))} px,
        {fmt((comp.get('region') or {}).get('fraction_of_frame'), 5)} of the frame,
        feather {esc((comp.get('region') or {}).get('feather_px'))} px</td></tr>
      <tr><td>Donor provenance</td><td><code>{esc(don.get('path'))}</code><br>
        sha256 <code>{esc((don.get('sha256') or '')[:32])}&hellip;</code><br>
        frame policy {esc(don.get('frame_policy'))}</td></tr>
    </table>

    <h3>Measurements</h3>
    <table class="data">
      <thead><tr><th>Arm</th><th>NCC vs open</th><th>NCC vs closed</th>
        <th>Gradient energy in region</th><th>PSNR outside region, dB</th>
        <th>Seam step / control</th><th>Region movement</th></tr></thead>
      <tbody>{''.join(measure_rows) or '<tr><td colspan="7">not measured</td></tr>'}</tbody>
    </table>
    <p class="note">NCC columns are a <b>proxy</b>: the region's correlation with this
      case's own open and closed references. Higher correlation with the closed reference
      is evidence, not a panel-state detector. Gradient energy near zero would mean the
      region was flattened into a patch; the source region measures
      {fmt((case.get('evaluation', {}).get('region_texture_references') or {})
           .get('source_region_gradient'))} and the donor region
      {fmt((case.get('evaluation', {}).get('region_texture_references') or {})
           .get('donor_region_gradient'))}.</p>

    {('<h3>Classifier scores</h3><table class="data"><thead><tr><th>Arm</th>'
      '<th>P(Closed Panel Cover)</th><th>Predicted index</th></tr></thead><tbody>'
      + ''.join(cls_rows) + '</tbody></table><p class="warn">'
      + esc(cls.get('_warning', '')) + '</p>') if cls_rows else ''}

    {('<h3>Overlays</h3><div class="shots">' + ''.join(
        f'<figure><img src="{esc(f)}" alt="{esc(f)}"><figcaption>{esc(Path(f).name)}'
        '</figcaption></figure>' for f in sorted(overlays)) + '</div>') if overlays else ''}
  </section>""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panel-cover review package</title>
<style>
  :root {{ --bg:#faf9f7; --fg:#1b1b1b; --muted:#5d5d5d; --line:#d8d4cd; --card:#fff;
           --warn:#8a3b00; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#15171a; --fg:#e9e7e3; --muted:#a3a097; --line:#2e3238; --card:#1c1f23;
             --warn:#e0a060; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.55 -apple-system,
          BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
          padding:0 16px 64px; }}
  main {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:1.6rem; margin:28px 0 4px; }}
  h2 {{ font-size:1.25rem; margin:0 0 4px; }}
  h3 {{ font-size:1rem; margin:22px 0 6px; text-transform:uppercase;
        letter-spacing:.06em; color:var(--muted); }}
  .lede, .pair, .note, .warn {{ color:var(--muted); }}
  .warn {{ color:var(--warn); }}
  .case {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
           padding:18px; margin:22px 0; }}
  .controls {{ display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 14px; }}
  button {{ font:inherit; padding:6px 12px; border:1px solid var(--line);
            border-radius:6px; background:transparent; color:var(--fg); cursor:pointer; }}
  .arms {{ display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); }}
  .arm {{ margin:0; }}
  .arm figcaption {{ display:flex; flex-direction:column; gap:2px; margin-bottom:6px;
                     font-size:.86rem; }}
  .arm figcaption span {{ color:var(--muted); }}
  video.clip {{ width:100%; border-radius:6px; background:#000; }}
  table {{ border-collapse:collapse; width:100%; font-size:.88rem; }}
  th, td {{ border:1px solid var(--line); padding:5px 8px; text-align:left;
            vertical-align:top; }}
  th {{ font-weight:600; }}
  table.kv td:first-child {{ width:30%; color:var(--muted); }}
  .shots {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }}
  .shots img {{ width:100%; border:1px solid var(--line); border-radius:6px; }}
  .shots figcaption {{ font-size:.8rem; color:var(--muted); }}
  code {{ font-size:.85em; word-break:break-all; }}
  ul {{ padding-left:20px; }}
</style>
</head>
<body>
<main>
  <h1>Panel-cover counterfactual: local review package</h1>
  <p class="lede">Generated {esc(meta.get('generated'))}. Class 2 <i>Opened Panel Cover</i>
    edited toward class 6 <i>Closed Panel Cover</i>. Every clip below is a
    <b>candidate</b>: nothing here has passed a completed human review form, and a closed
    cover is not evidence that a scene is safe.</p>

  <h3>How to read this</h3>
  <ul>
    <li>The intervention is a <b>region replacement</b>. The panel region of the source
      holds registered pixels from a donor clip in which the cover is shut. <b>No closing
      motion was generated.</b></li>
    <li>Compositing and refinement are judged separately. Compositing can succeed while
      refinement adds nothing or makes things worse.</li>
    <li><b>refined</b> keeps the whole generated frame. <b>refined_restricted</b> puts the
      region back over the composite afterwards. That is output compositing, not
      mask-conditioned generation.</li>
    <li>Classifier scores, where present, are reported apart from the picture
      measurements and prove nothing about the panel.</li>
  </ul>
  {''.join(rows)}
  <h3>Files</h3>
  <p class="lede">Manifests, settings, decisions and logs are in this directory.
    <code>SHA256SUMS.txt</code> covers every file. Model weights and dataset clips are
    deliberately not included.</p>
</main>
<script>
  function clips(id) {{
    return Array.prototype.slice.call(
      document.querySelectorAll('video.clip[data-case="' + id + '"]'));
  }}
  document.addEventListener('click', function (e) {{
    var t = e.target;
    if (t.dataset.play) {{
      clips(t.dataset.play).forEach(function (v) {{ v.currentTime = 0; v.play(); }});
    }} else if (t.dataset.pause) {{
      clips(t.dataset.pause).forEach(function (v) {{ v.pause(); }});
    }} else if (t.dataset.reset) {{
      clips(t.dataset.reset).forEach(function (v) {{ v.pause(); v.currentTime = 0; }});
    }}
  }});
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="configs/panel_cover_plan.json")
    ap.add_argument("--stage", default="/ocean/projects/cis260288p/puprety/transfer/"
                                       "panel-cover-review")
    ap.add_argument("--mac-dest", default="/Users/prashantuprety/workspace/"
                                          "unsafenet-counterfactuals/unsafenet_cf_outputs/"
                                          "panel-cover-review")
    ap.add_argument("--ssh-user", default=None)
    args = ap.parse_args()

    stage = Path(args.stage)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    plan = json.loads((REPO / args.plan).read_text())
    evaluation = {}
    ev_path = OUT / "evaluation.json"
    if ev_path.exists():
        evaluation = json.loads(ev_path.read_text()).get("cases", {})

    cases = []
    for case in plan["cases"]:
        cid = case["case_id"]
        files: list[str] = []
        staged_sources: set[Path] = set()
        for key, _, _ in ARMS:
            # The guidance arms are written to <case>/guided/, every other arm to
            # <case>/videos/. Both are staged under the one name the page looks for.
            src = OUT / cid / "videos" / f"{cid}_{key}.mp4"
            if not src.exists():
                src = OUT / cid / "guided" / f"{cid}_{key}.mp4"
            rel = f"videos/{cid}/{cid}_{key}.mp4"
            if copy(src, stage / rel):
                files.append(rel)
                staged_sources.add(src.resolve())
        for p in sorted((OUT / cid / "overlays").glob("*.png")) if (
                OUT / cid / "overlays").exists() else []:
            rel = f"overlays/{cid}/{p.name}"
            copy(p, stage / rel)
            files.append(rel)
        for p in sorted((OUT / cid / "guided").glob("*.mp4")) if (
                OUT / cid / "guided").exists() else []:
            if p.resolve() in staged_sources:
                continue        # already staged above under its arm name
            rel = f"videos/{cid}/guided_{p.name}"
            copy(p, stage / rel)
            files.append(rel)
        comp_path = OUT / cid / "composite.json"
        comp = json.loads(comp_path.read_text()) if comp_path.exists() else {}
        if comp_path.exists():
            copy(comp_path, stage / f"manifests/{cid}_composite.json")
        cases.append({"case_id": cid, "source_clip": case.get("source_clip"),
                      "donor_clip": case.get("donor_clip"), "files": files,
                      "composite": comp, "evaluation": evaluation.get(cid, {})})
        log(f"{cid}: staged {len(files)} media files")

    # Documents, manifests, logs.
    for rel in ("DECISIONS.md", "docs/PANEL_COVER_DEFINITIONS.md",
                "docs/ACCEPTANCE_CRITERIA.md", "docs/GROUPING_CALIBRATION.md",
                "docs/SU_LEDGER.md", "templates/REVIEW_TEMPLATE.md",
                "configs/panel_cover_plan.json", "configs/panel_cover_generate.json",
                "configs/panel_cover_regions.json", "configs/panel_cover_guided.json",
                "outputs/panel_cover/FINDINGS.md"):
        copy(REPO / rel, stage / Path(rel).name)
    for rel in ("generate_manifest.json", "composite_manifest.json", "evaluation.json",
                "diagnostics.json", "classifier/training_report.json",
                "classifier/label_map.json", "audit/audit.json"):
        copy(OUT / rel, stage / "manifests" / Path(rel).name)
    for p in sorted(REPO.glob("logs/pc*-*.out")) + sorted(REPO.glob("logs/pc*-*.err")):
        copy(p, stage / "logs" / p.name)
    for p in sorted((OUT / "audit" / "diffs").glob("*.png")) if (
            OUT / "audit" / "diffs").exists() else []:
        copy(p, stage / "audit_diffs" / p.name)

    meta = {"generated": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(timespec="seconds")}
    (stage / "index.html").write_text(build_page(cases, meta))

    sums = []
    for p in sorted(stage.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            sums.append(f"{sha256(p)}  {p.relative_to(stage)}")
    (stage / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")

    user = args.ssh_user or __import__("os").environ.get("USER", "<user>")
    (stage / "HOW_TO_DOWNLOAD.txt").write_text(f"""Panel-cover review package
==========================

Run these two commands on the Mac, not on Bridges-2.

  mkdir -p "{args.mac_dest}"
  rsync -avh --progress \\
    {user}@bridges2.psc.edu:{stage}/ \\
    "{args.mac_dest}/"

Then open the page:

  open "{args.mac_dest}/index.html"

Verify the transfer from inside the downloaded directory:

  cd "{args.mac_dest}" && shasum -a 256 -c SHA256SUMS.txt

The page uses relative paths only, so it works offline once the directory is local.

The clips show identifiable factory workers who consented to the dataset authors' study,
not to redistribution. Keep this package local.
""")
    log(f"\nstaged {len(sums)} files at {stage}")
    log(f"page: {stage / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
