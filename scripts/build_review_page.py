#!/usr/bin/env python3
"""Build a local HTML review page gathering every artefact of the experiment.

A single file opened in a browser, with the videos playable side by side and the numbers
next to them. It is written to the outputs tree and is meant to be rsynced to a laptop
along with the clips it references, which it links by relative path.

The page states what each number does and does not establish. It carries no signature
field that can be ticked by accident: the playback review stays unsigned until a human
completes templates/REVIEW_TEMPLATE.md.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def gate_badge(passed) -> str:
    if passed is None:
        return '<span class="badge pending">pending</span>'
    return ('<span class="badge pass">PASS</span>' if passed
            else '<span class="badge fail">FAIL</span>')


def video_cell(label: str, rel: str | None) -> str:
    if rel is None:
        return f"<figure><figcaption>{html.escape(label)}</figcaption>" \
               f"<p class='missing'>not generated</p></figure>"
    return (f"<figure><figcaption>{html.escape(label)}</figcaption>"
            f"<video src='{html.escape(rel)}' controls loop muted playsinline></video>"
            f"</figure>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "outputs" / "review" ))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    (out_dir / "media").mkdir(parents=True, exist_ok=True)

    clf = REPO / "outputs" / "classifier"
    gud = REPO / "outputs" / "guided"

    audit = read_json(clf / "splits.json")
    groups = read_json(clf / "groups.json")
    train = read_json(clf / "train_result.json")
    probe_train = read_json(clf / "probe_train.json")
    evalr = read_json(clf / "eval_report.json")
    probe_g = read_json(gud / "probe_guidance.json")
    guided = read_json(gud / "manifest_guided.json")
    label_map = read_json(clf / "label_map.json")

    # Copy the media the page references, so the directory is self-contained to transfer.
    media: dict[str, str | None] = {}
    candidates = {
        "source (preprocessed)": REPO / "outputs" / "preprocessed" /
            "0_tr143_720x480x49.mp4",
        "VAE reconstruction": REPO / "outputs" / "stage3_vae" / "reconstruction.mp4",
    }
    if guided:
        o = guided.get("outputs", {})
        candidates["unguided control"] = Path(o.get("unguided_control_clip", "/nonexistent"))
        candidates["classifier-guided candidate"] = Path(o.get("guided_clip", "/nonexistent"))
    for label, src in candidates.items():
        if src.exists():
            dest = out_dir / "media" / src.name
            shutil.copy2(src, dest)
            media[label] = f"media/{src.name}"
        else:
            media[label] = None

    sheets = []
    for src in [gud / "comparison_guided_vs_unguided.png",
                REPO / "outputs" / "stage3_vae" / "source_vs_reconstruction.png"]:
        if src.exists():
            shutil.copy2(src, out_dir / "media" / src.name)
            sheets.append((src.name, f"media/{src.name}"))

    gates = {
        "A — data integrity": None if audit is None else True,
        "B0 — training memory probe": None if probe_train is None else True,
        "B1 — validation balanced accuracy >= 0.80":
            None if train is None else train["gates"]["B1_balanced_accuracy_at_least_0.80"],
        "B2 — each per-class recall >= 0.75":
            None if train is None else train["gates"]["B2_each_recall_at_least_0.75"],
        "B3 — original vs VAE consistency":
            None if evalr is None else evalr["gates"]["B3_vae_consistency"],
        "C — shortcut diagnostics":
            None if evalr is None else evalr["gates"]["C_shortcut_diagnostics"],
        "D — guided-step gradient probe":
            None if probe_g is None else all(
                v for k, v in probe_g["gates"].items() if isinstance(v, bool)),
        "E — guided candidate generated": None if guided is None else True,
    }

    rows_gates = "\n".join(
        f"<tr><td>{html.escape(k)}</td><td>{gate_badge(v)}</td></tr>"
        for k, v in gates.items())

    # Classifier probabilities
    prob_rows = ""
    if guided:
        probs = guided["outputs"].get("classifier_probabilities_p_safe", {})
        psnr = guided["outputs"].get("psnr_vs_source_db", {})
        for k, v in probs.items():
            prob_rows += (f"<tr><td>{html.escape(k)}</td><td>{fmt(v)}</td>"
                          f"<td>{fmt(psnr.get(k))}</td></tr>")

    mask_rows = ""
    if evalr:
        for name, r in evalr["masks"].items():
            mask_rows += (f"<tr><td>{html.escape(name)}</td>"
                          f"<td>{fmt(r['balanced_accuracy'])}</td>"
                          f"<td>{fmt(r['drop_from_baseline'])}</td>"
                          f"<td>{r['prediction_flips']}</td></tr>")

    videos = "\n".join(video_cell(k, v) for k, v in media.items())
    sheet_html = "\n".join(
        f"<figure><figcaption>{html.escape(n)}</figcaption>"
        f"<a href='{p}'><img src='{p}' alt='{html.escape(n)}'></a></figure>"
        for n, p in sheets)

    excluded = (groups or {}).get("excluded_from_all_splits", {})
    excluded_json = html.escape(json.dumps(excluded, indent=2))
    label_map_json = html.escape(
        json.dumps(label_map, indent=2) if label_map else "not written yet")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Walkway counterfactual review</title>
<style>
 :root {{ --fg:#1a1a1a; --bg:#fff; --muted:#666; --line:#ddd;
          --pass:#1a7f37; --fail:#b42318; --pending:#8a6d1f; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --fg:#e8e8e8; --bg:#131313; --muted:#9a9a9a; --line:#333;
            --pass:#3fb950; --fail:#f85149; --pending:#d4a72c; }} }}
 body {{ font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         color:var(--fg); background:var(--bg); margin:0 auto; padding:16px;
         max-width:1100px; }}
 h1 {{ font-size:1.6rem; margin:0 0 .2em; }}
 h2 {{ font-size:1.15rem; margin:2em 0 .4em; border-bottom:1px solid var(--line);
       padding-bottom:.3em; }}
 table {{ border-collapse:collapse; width:100%; margin:.5em 0 1em; }}
 th,td {{ text-align:left; padding:.45em .6em; border-bottom:1px solid var(--line);
          font-variant-numeric:tabular-nums; }}
 th {{ color:var(--muted); font-weight:600; font-size:.85rem;
       text-transform:uppercase; letter-spacing:.03em; }}
 .badge {{ font-size:.8rem; font-weight:700; padding:.15em .5em; border-radius:4px; }}
 .pass {{ color:var(--pass); }} .fail {{ color:var(--fail); }}
 .pending {{ color:var(--pending); }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
          gap:1rem; }}
 figure {{ margin:0; }} figcaption {{ font-size:.85rem; color:var(--muted);
          margin-bottom:.3em; }}
 video,img {{ width:100%; border:1px solid var(--line); border-radius:6px; }}
 .note {{ border-left:3px solid var(--line); padding:.1em 0 .1em 1em;
          color:var(--muted); }}
 .warn {{ border-left-color:var(--fail); }}
 .missing {{ color:var(--muted); font-style:italic; }}
 code {{ font-size:.9em; }}
</style></head><body>

<h1>Walkway counterfactual — review</h1>
<p class="note">Source clip <code>0_tr143.mp4</code>, dataset class 0
<em>Safe Walkway Violation</em>, target class 4 <em>Safe Walkway</em>.
Everything below is a <strong>candidate counterfactual pending human review</strong>.</p>

<p class="note warn"><strong>A classifier prediction flip is not proof of corrected
behaviour.</strong> The absence of a violation is not the presence of safe behaviour. The
playback review is <strong>unsigned</strong>: complete
<code>templates/REVIEW_TEMPLATE.md</code> against the clips before drawing any conclusion.
Official test-split evaluation is <strong>PENDING</strong> by decision; every number here
comes from training and validation data only.</p>

<h2>Gates</h2>
<table><tr><th>Gate</th><th>Result</th></tr>
{rows_gates}
</table>

<h2>Clips</h2>
<div class="grid">
{videos}
</div>

<h2>Classifier probabilities and fidelity</h2>
<p class="note">p(safe walkway) is the classifier's output for binary index 1. PSNR is
measured against the preprocessed source; the VAE round trip sets the ceiling no edit can
exceed.</p>
<table><tr><th>Clip</th><th>p(safe walkway)</th><th>PSNR vs source (dB)</th></tr>
{prob_rows or '<tr><td colspan="3" class="missing">not generated yet</td></tr>'}
</table>

<h2>Shortcut diagnostics</h2>
<p class="note">Three masks of identical area. The <strong>control</strong> mask covers a
label-irrelevant region: if its drop matches the worker or lane drop, masking itself is
moving the prediction and neither diagnostic supports a conclusion. Both worker and lane
carry information the decision needs, so a drop under either is expected. The diagnosis is
the pattern across all three, not one number.</p>
<table><tr><th>Mask</th><th>Balanced accuracy</th><th>Drop</th><th>Flips</th></tr>
{mask_rows or '<tr><td colspan="4" class="missing">not evaluated yet</td></tr>'}
</table>

<h2>Comparison sheets</h2>
<div class="grid">
{sheet_html or '<p class="missing">none yet</p>'}
</div>

<h2>Label map</h2>
<pre><code>{label_map_json}</code></pre>

<h2>Recording-group isolation</h2>
<p class="note">Clips excluded from every split because they share a recording, or the
scene, with the source clip. Frame overlap is treated as certain evidence of one recording;
a similar background alone is not.</p>
<pre><code>{excluded_json}</code></pre>

</body></html>
"""
    (out_dir / "index.html").write_text(page)
    print(f"wrote {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
