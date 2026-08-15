#!/usr/bin/env python3
"""Build an episode review manifest and offline timestamp-labeling page.

Input may be a JSON list, ``{"episodes": [...]}``, or JSONL. The generated
page never uploads data: reviewers can add event rows and download JSONL.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ALLOWED_LABELS = ("productive", "stall", "regress", "recover", "hesitate", "abandon", "complete", "other")


def _safe_media_reference(value: Any) -> tuple[str, str]:
    """Never propagate likely signed URLs/tokens into generated artifacts."""
    reference = str(value or "")
    lowered = reference.lower()
    parsed = urlparse(reference)
    sensitive_markers = ("x-amz-", "signature=", "credential=", "token=", "secret=", "access_key")
    if any(marker in lowered for marker in sensitive_markers) or (parsed.scheme in {"http", "https"} and parsed.query):
        return "", "Remote/signed media reference omitted; review it in the authorized external browser."
    return reference, ""


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        if path.suffix.lower() == ".zarr":
            return [{"episode_id": path.stem, "task": "unknown", "episode_path": str(path), "video_path": ""}]
        videos = []
        for suffix in ("*.mp4", "*.mov", "*.webm", "*.mkv"):
            videos.extend(path.rglob(suffix))
        return [{"episode_id": video.stem, "task": "unknown", "video_path": str(video)} for video in sorted(videos)]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    elif path.suffix.lower() not in {".json", ".jsonl"}:
        return [{"episode_id": path.stem, "task": "unknown", "episode_path": str(path), "video_path": str(path) if path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"} else ""}]
    elif path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("episodes", [payload]) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("metadata must be a JSON list, {'episodes': [...]}, or JSONL")
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        generic_path = record.get("path")
        generic_is_zarr = str(generic_path or "").lower().endswith(".zarr")
        video, video_note = _safe_media_reference(record.get("video_path") or record.get("video") or ("" if generic_is_zarr else generic_path))
        episode_path, path_note = _safe_media_reference(record.get("episode_path") or record.get("zarr_path") or (generic_path if generic_is_zarr else ""))
        normalized.append(
            {
                "episode_id": str(record.get("episode_id") or record.get("id") or f"episode_{index:03d}"),
                "task": str(record.get("task") or record.get("task_description") or "unknown"),
                "video_path": str(video),
                "duration_sec": record.get("duration_sec"),
                **({"episode_path": episode_path} if episode_path else {}),
                **({"review_note": video_note or path_note} if video_note or path_note else {}),
            }
        )
    return normalized


def write_html(episodes: list[dict[str, Any]], output: Path) -> None:
    data = json.dumps(episodes).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    options = "".join(f'<option value="{label}">{label}</option>' for label in ALLOWED_LABELS)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>EgoFlow manual review</title><style>
:root{{--bg:#0d1420;--panel:#161f2e;--text:#ebf0f8;--muted:#9ca8bb;--accent:#42d3bd}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:24px}} h1{{margin:0 0 5px}} .muted{{color:var(--muted)}}
.bar,.card{{background:var(--panel);padding:16px;border-radius:10px;margin:15px 0}} select,input,button{{padding:8px;background:#202c40;color:var(--text);border:1px solid #46536a;border-radius:5px}}
button{{cursor:pointer}} button.primary{{background:var(--accent);color:#071713;font-weight:700}} video{{display:block;width:100%;max-height:540px;background:#000;margin:12px 0}}
.form{{display:grid;grid-template-columns:1fr 1fr 1.2fr 3fr auto;gap:8px}} table{{width:100%;border-collapse:collapse}}td,th{{padding:7px;border-bottom:1px solid #334057;text-align:left}}
@media(max-width:760px){{.form{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><h1>EgoFlow manual event review</h1>
<p class="muted">Offline helper. Labels remain in this browser until you download JSONL. Review labels validate predictions; they are not automatic model outputs.</p>
<div class="bar"><label>Episode <select id="episode"></select></label> <button id="previous">Previous</button> <button id="next">Next</button>
<strong id="task"></strong><div id="location" class="muted"></div></div>
<div class="card"><video id="video" controls preload="metadata"></video><div class="form">
<input id="start" type="number" step="0.1" min="0" placeholder="start seconds"><input id="end" type="number" step="0.1" min="0" placeholder="end seconds">
<select id="label">{options}</select><input id="note" placeholder="brief observable note"><button class="primary" id="add">Add event</button></div>
<p><button id="startNow">Start = video time</button> <button id="endNow">End = video time</button> <button id="download">Download manual_labels.jsonl</button></p></div>
<div class="card"><table><thead><tr><th>Episode</th><th>Time</th><th>Label</th><th>Note</th><th></th></tr></thead><tbody id="rows"></tbody></table></div>
<script>const episodes={data}; const labels=[]; const ep=document.querySelector('#episode'), video=document.querySelector('#video');
episodes.forEach((e,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=`${{e.episode_id}} / ${{e.task}}`;ep.appendChild(o)}});
function show(i){{if(!episodes.length)return;i=Math.max(0,Math.min(episodes.length-1,i));ep.value=i;const e=episodes[i];document.querySelector('#task').textContent=e.task;document.querySelector('#location').textContent=e.video_path||e.episode_path||e.review_note||'No local video: review in the external browser and type timestamps.';if(e.video_path){{video.src=e.video_path;video.style.display='block'}}else{{video.removeAttribute('src');video.style.display='none'}}}}
function render(){{const body=document.querySelector('#rows');body.innerHTML='';labels.forEach((x,i)=>{{const r=body.insertRow();[x.episode_id,`${{x.start_sec.toFixed(1)}}-${{x.end_sec.toFixed(1)}}`,x.label,x.note].forEach(v=>r.insertCell().textContent=v);const c=r.insertCell(),b=document.createElement('button');b.textContent='remove';b.onclick=()=>{{labels.splice(i,1);render()}};c.appendChild(b)}})}}
ep.onchange=()=>show(+ep.value);document.querySelector('#previous').onclick=()=>show(+ep.value-1);document.querySelector('#next').onclick=()=>show(+ep.value+1);
document.querySelector('#startNow').onclick=()=>document.querySelector('#start').value=video.currentTime.toFixed(1);document.querySelector('#endNow').onclick=()=>document.querySelector('#end').value=video.currentTime.toFixed(1);
document.querySelector('#add').onclick=()=>{{const e=episodes[+ep.value],start=+document.querySelector('#start').value,end=+document.querySelector('#end').value;if(!e||!Number.isFinite(start)||!Number.isFinite(end)||end<start){{alert('Enter a valid start/end range.');return}}labels.push({{episode_id:e.episode_id,task:e.task,start_sec:start,end_sec:end,label:document.querySelector('#label').value,note:document.querySelector('#note').value}});render()}};
document.querySelector('#download').onclick=()=>{{const blob=new Blob([labels.map(x=>JSON.stringify(x)).join('\n')+'\n'],{{type:'application/x-ndjson'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='manual_labels.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};show(0);</script></main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, nargs="+", help="episode JSON/JSONL, Zarr path, or directory of videos")
    parser.add_argument("--manifest", type=Path, default=Path("review_manifest.json"))
    parser.add_argument("--output", type=Path, help="compatibility alias for --manifest")
    parser.add_argument("--html", type=Path, default=Path("review.html"))
    args = parser.parse_args()
    episodes = [episode for source in args.metadata for episode in load_metadata(source)]
    if args.output:
        args.manifest = args.output
    manifest = {"schema_version": "egoflow.review.v1", "episode_count": len(episodes), "episodes": episodes}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.suffix.lower() == ".jsonl":
        args.manifest.write_text("".join(json.dumps(episode) + "\n" for episode in episodes), encoding="utf-8")
    else:
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_html(episodes, args.html)
    print(f"review episodes: {len(episodes)}")
    print(f"manifest: {args.manifest}")
    print(f"offline review page: {args.html}")
    if not any(item.get("video_path") for item in episodes):
        print("note: no local videos supplied; use external browser paths and type timestamps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
