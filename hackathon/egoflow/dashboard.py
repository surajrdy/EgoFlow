"""Dependency-free presentation dashboard for an EgoFlow review database."""

from __future__ import annotations

from collections import defaultdict
import html
import json
import os
from pathlib import Path
import sqlite3


SOURCE_CLASS = {
    "LEARNED": "learned",
    "LEARNED V2": "learned-v2",
    "HYBRID": "hybrid",
    "AUX": "aux",
    "HAND EXPERIMENTAL": "hand",
}


def _sparkline(points: list[tuple[float, float, float]], duration: float) -> str:
    if not points:
        return ""
    line = " ".join(
        f"{100 * timestamp / max(duration, 1e-6):.2f},{44 - 38 * max(0.0, min(1.0, progress)):.2f}"
        for timestamp, progress, _ in points
    )
    return f"""
    <svg class="progress-chart" viewBox="0 0 100 48" preserveAspectRatio="none" role="img" aria-label="Learned task progress from zero to one hundred percent">
      <line x1="0" y1="6" x2="100" y2="6" class="chart-grid"/>
      <line x1="0" y1="25" x2="100" y2="25" class="chart-grid"/>
      <line x1="0" y1="44" x2="100" y2="44" class="chart-grid"/>
      <polyline points="{line}"/>
    </svg>"""


def render_dashboard(
    database_path: str | Path,
    output_path: str | Path,
    *,
    hero_image: str | Path | None = None,
    hero_video: str | Path | None = None,
    featured_episode_ids: list[str] | None = None,
    include_research_events: bool = False,
) -> Path:
    """Render a restrained, self-contained research presentation from SQLite."""

    database, output = Path(database_path), Path(output_path)
    connection = sqlite3.connect(database)
    episodes = connection.execute(
        "SELECT episode_id,task,duration_sec,completion_score,video_path FROM episodes ORDER BY task,episode_id"
    ).fetchall()
    if featured_episode_ids:
        featured_order = {episode_id: index for index, episode_id in enumerate(featured_episode_ids)}
        episodes = [row for row in episodes if row[0] in featured_order]
        episodes.sort(key=lambda row: featured_order[row[0]])
    events_by_episode: dict[str, list[tuple]] = defaultdict(list)
    for row in connection.execute(
        "SELECT episode_id,start_sec,end_sec,label,source,confidence,reason FROM events ORDER BY start_sec"
    ):
        events_by_episode[row[0]].append(row[1:])
    clean_by_episode: dict[str, list[tuple]] = defaultdict(list)
    for row in connection.execute(
        "SELECT episode_id,start_sec,end_sec,duration_sec FROM clean_spans ORDER BY start_sec"
    ):
        clean_by_episode[row[0]].append(row[1:])
    manual_by_episode: dict[str, list[tuple]] = defaultdict(list)
    for row in connection.execute(
        "SELECT episode_id,start_sec,end_sec,label,note FROM manual_events ORDER BY start_sec"
    ):
        manual_by_episode[row[0]].append(row[1:])
    progress_by_episode: dict[str, list[tuple]] = defaultdict(list)
    for row in connection.execute(
        "SELECT episode_id,timestamp_sec,progress,rate FROM progress_points ORDER BY timestamp_sec"
    ):
        progress_by_episode[row[0]].append(row[1:])
    documents = {
        name: json.loads(payload)
        for name, payload in connection.execute("SELECT name,payload_json FROM run_documents")
    }
    connection.close()

    def relative_asset(value: str | Path | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        absolute = path if path.is_absolute() else Path.cwd() / path
        return Path(os.path.relpath(absolute.resolve(), output.parent.resolve())).as_posix()

    total_seconds = sum(row[2] for row in episodes)
    event_total = sum(
        1 for rows in events_by_episode.values() for row in rows if row[3] == "LEARNED V2"
    )
    clean_total = sum(span[2] for rows in clean_by_episode.values() for span in rows)
    manual_total = sum(len(rows) for rows in manual_by_episode.values())
    cards: list[str] = []
    for episode_id, task, duration, completion, video_path in episodes:
        # The presentation surface stays intentionally narrow: learned hand-
        # dynamics candidates only. Legacy derived/auxiliary layers remain in
        # SQLite and result JSON for auditability, not in the live demo.
        events = [
            event for event in events_by_episode[episode_id]
            if str(event[2]).lower() not in {"productive", "stall"}
            and (include_research_events or event[3] == "LEARNED V2")
        ]
        manual = manual_by_episode[episode_id]
        clean = clean_by_episode[episode_id]
        local_video_href = relative_asset(video_path)
        video_href = local_video_href or f"https://partners.mecka.ai/api/egoverse/uploads/{episode_id}/video?redirect=1"
        escaped_video = html.escape(video_href, quote=True)
        event_marks = "".join(
            f'<span class="event-mark {SOURCE_CLASS.get(source, "unknown")}" data-source="{SOURCE_CLASS.get(source, "unknown")}" '
            f'style="left:{100*start/max(duration,1e-6):.2f}%;width:{max(.45,100*(end-start)/max(duration,1e-6)):.2f}%" '
            f'title="{html.escape(label.replace("_", " "))} · {html.escape(source)} · {start:.2f}–{end:.2f}s" '
            f'tabindex="0" role="button" data-action="watch" data-video="{escaped_video}" data-start="{start:.3f}" '
            f'data-end="{end:.3f}" data-episode="{html.escape(episode_id, quote=True)}" data-confidence="{confidence:.4f}" '
            f'data-title="{html.escape(label.replace("_", " ").title(), quote=True)} · {html.escape(episode_id, quote=True)}"></span>'
            for start, end, label, source, confidence, reason in events
        )
        manual_marks = "".join(
            f'<span class="manual-mark" data-source="human" style="left:{100*start/max(duration,1e-6):.2f}%;'
            f'width:{max(.45,100*(end-start)/max(duration,1e-6)):.2f}%" '
            f'title="human {html.escape(label)} · {start:.2f}–{end:.2f}s" tabindex="0" role="button" '
            f'data-action="watch" data-video="{escaped_video}" data-start="{start:.3f}" data-end="{end:.3f}" '
            f'data-title="Human: {html.escape(label.replace("_", " ").title(), quote=True)} · {html.escape(episode_id, quote=True)}"></span>'
            for start, end, label, note in manual
        )
        rows = "".join(
            f'<tr data-row-source="{SOURCE_CLASS.get(source,"unknown")}"><td><button class="time-link" data-action="watch" '
            f'data-video="{escaped_video}" data-start="{start:.3f}" data-end="{end:.3f}" '
            f'data-title="{html.escape(label.replace("_", " ").title(), quote=True)} · {html.escape(episode_id, quote=True)}">{start:.2f}–{end:.2f}s</button></td>'
            f'<td>{html.escape(label.replace("_", " ").upper())}</td><td>{html.escape(source)}</td><td>{confidence:.2f}</td></tr>'
            for start, end, label, source, confidence, reason in events
        ) or '<tr><td colspan="4" class="muted">No review candidates</td></tr>'
        clean_html = "".join(
            f'<span class="clean-segment">{start:.1f}–{end:.1f}s <b>{length:.1f}s</b></span>'
            for start, end, length in clean
        ) or '<span class="muted">No candidate-free span above the threshold.</span>'
        video_note = "cached" if local_video_href else "public stream"
        flags = {SOURCE_CLASS.get(row[3], "unknown") for row in events}
        if manual:
            flags.add("human")
        cards.append(f"""
        <article class="episode" data-testid="episode-card" data-id="{html.escape(episode_id, quote=True)}" data-task="{html.escape(task.replace('_',' '), quote=True)}" data-search="{html.escape((episode_id+' '+task).lower())}" data-flags="{' '.join(flags)}">
          <header class="episode-head">
            <div><p class="kicker">{html.escape(task.replace('_',' '))}</p><code>{episode_id}</code></div>
            <div class="score"><b>{100*(completion or 0):.1f}%</b><span>completion score</span></div>
          </header>
          <div class="chart-heading"><span>Learned task progress</span><span>0 → 100%</span></div>
          <div class="chart-wrap"><span class="y y100">100</span><span class="y y50">50</span><span class="y y0">0</span>{_sparkline(progress_by_episode[episode_id], duration)}</div>
          <div class="track-row"><span>Review windows</span><div class="event-track">{event_marks}</div></div>
          <div class="track-row human-row"><span>Human labels</span><div class="event-track human-track">{manual_marks}</div></div>
          <div class="axis"><span>0s</span><span>{duration:.1f}s</span></div>
          <footer><span>{duration/60:.1f} min</span><span>{len(events)} indexed windows</span><span>{len(clean)} clean spans</span><button class="watch" data-action="watch" data-video="{escaped_video}" data-start="0" data-title="{html.escape(task.replace('_',' ').title(), quote=True)} · {html.escape(episode_id, quote=True)}">Watch episode · {video_note}</button></footer>
          <details><summary>Intervals and export spans</summary>
            <div class="detail-grid"><div><table><thead><tr><th>Window</th><th>Candidate</th><th>Source</th><th>Conf.</th></tr></thead><tbody>{rows}</tbody></table></div>
            <div><h4>Candidate-free segments ≥10 seconds</h4><div class="clean-list">{clean_html}</div></div></div>
          </details>
        </article>""")

    hero_image_href = relative_asset(hero_image)
    hero_video_href = relative_asset(hero_video)
    poster = f' poster="{html.escape(hero_image_href)}"' if hero_image_href else ""
    if hero_video_href:
        hero_media = f'<video controls playsinline preload="metadata"{poster} src="{html.escape(hero_video_href)}"></video>'
    elif hero_image_href:
        hero_media = f'<img src="{html.escape(hero_image_href)}" alt="EgoFlow Angry Bird result">'
    else:
        hero_media = '<div class="media-placeholder">Run the demo renderer to attach the scored video.</div>'
    interaction = documents.get("interaction_v2_validation", {})
    learned_event = interaction.get("hero_validation_event", {}) if isinstance(interaction, dict) else {}
    hand = documents.get("hand_v2_validation", {})
    hand_event = hand.get("hero_candidate", {}) if isinstance(hand, dict) else {}
    metric_summary = html.escape(json.dumps(documents.get("blind_test_metrics", {}), indent=2)[:2200])
    if include_research_events:
        evidence_options = """<option value="all">All candidate types</option><option value="learned-v2">Learned v2</option><option value="hybrid">Hybrid</option><option value="aux">Auxiliary</option><option value="hand">Hand geometry</option><option value="learned">Progress-derived</option><option value="human">Reviewed spans</option>"""
        initial_layer_note = "All sparse candidate types are visible. Click any colored interval to inspect that scored moment."
    else:
        evidence_options = """<option value="demo">Learned + reviewed</option><option value="learned-v2">Learned only</option><option value="human">Reviewed spans only</option>"""
        initial_layer_note = "Learned interaction candidates and reviewed spans. Click any timeline mark to inspect that moment."

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EgoFlow — demonstration review</title>
<style>
:root {{ --paper:#f5f4ef; --white:#fff; --ink:#17202a; --muted:#64707d; --rule:#d5d8d8; --blue:#2563a6; --teal:#16796c; --alert:#c33d48; --orange:#a55e18; --rose:#a43c67; --violet:#7651a8; --gray:#8c939a; }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
main {{ max-width:1240px; margin:auto; padding:28px 38px 80px }} h1,h2,h3,p {{ margin-top:0 }} h1 {{ font:600 40px/1.08 Georgia,serif; letter-spacing:-.025em; max-width:720px }} h2 {{ font:600 27px/1.2 Georgia,serif }} h3 {{ font-size:17px }} a {{ color:var(--blue); text-underline-offset:3px }} code {{ font:11px ui-monospace,SFMono-Regular,monospace; color:var(--muted) }}
.topline {{ display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--ink); padding-bottom:12px; margin-bottom:38px }} .brand {{ font-weight:750; letter-spacing:.14em; font-size:12px }} .version {{ color:var(--muted); font-size:12px }} .intro {{ display:grid; grid-template-columns:1.5fr .7fr; gap:70px; margin-bottom:35px }} .intro p {{ color:#46525e; font-size:17px; max-width:700px }} .claim {{ border-left:3px solid var(--teal); padding-left:18px; align-self:end }} .claim b {{ display:block; font-size:23px }} .claim span {{ color:var(--muted); font-size:13px }}
.stats {{ display:grid; grid-template-columns:repeat(5,1fr); border-block:1px solid var(--rule); margin-bottom:42px }} .stat {{ padding:15px 18px; border-right:1px solid var(--rule) }} .stat:last-child {{ border:0 }} .stat b {{ display:block; font-size:25px }} .stat span {{ color:var(--muted); font-size:12px }}
.hero {{ display:grid; grid-template-columns:1.55fr .7fr; gap:24px; margin-bottom:54px }} .hero-media {{ background:#111; border:1px solid #222 }} .hero-media video,.hero-media img {{ display:block; width:100%; aspect-ratio:16/10.3; object-fit:contain }} .media-placeholder {{ min-height:460px; display:grid; place-items:center; color:#aaa }} .hero-copy {{ background:var(--white); border:1px solid var(--rule); padding:26px }} .overline {{ margin:0 0 8px; color:var(--teal); font-size:11px; font-weight:750; letter-spacing:.12em; text-transform:uppercase }} .result {{ border-block:1px solid var(--rule); padding:16px 0; margin:20px 0 }} .result b {{ display:block; font:600 28px Georgia,serif }} .result span {{ color:var(--muted) }} .legend {{ display:grid; grid-template-columns:1fr 1fr; gap:9px 14px; margin-top:22px; font-size:12px }} .key::before {{ content:""; display:inline-block; width:9px; height:9px; margin-right:7px; background:var(--gray) }} .key.learned-v2::before {{ background:var(--alert) }} .key.learned::before {{ background:var(--teal) }} .key.human::before {{ background:#aeb5b8 }}
.method {{ margin:0 0 55px }} .section-intro {{ max-width:720px; color:var(--muted) }} .flow {{ display:grid; grid-template-columns:1fr 30px 1.35fr 30px 1.1fr 30px 1fr; align-items:stretch; margin-top:24px }} .flow-node {{ background:var(--white); border:1px solid var(--rule); padding:19px; min-height:128px }} .flow-node b {{ display:block; margin-bottom:8px }} .flow-node span {{ color:var(--muted); font-size:13px }} .flow-arrow {{ display:grid; place-items:center; color:var(--muted); font-size:23px }} .branch {{ display:grid; grid-template-columns:1fr 1fr; gap:8px }} .branch div {{ border-left:2px solid var(--blue); padding-left:10px }} .branch div:last-child {{ border-color:var(--teal) }}
.reading {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; margin:0 0 56px }} .reading-panel {{ background:var(--white); border:1px solid var(--rule); padding:22px }} .mini-chart {{ width:100%; height:120px }} .mini-chart .base {{ stroke:#b7bcbc; stroke-width:1 }} .mini-chart .curve {{ stroke:var(--teal); fill:none; stroke-width:3 }} .mini-chart .spike {{ stroke:var(--alert); stroke-width:4 }} .mini-label {{ display:flex; justify-content:space-between; color:var(--muted); font-size:12px }}
.catalog-head {{ display:flex; justify-content:space-between; align-items:end; gap:20px; border-bottom:1px solid var(--ink); padding-bottom:14px }} .controls {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap }} input,select,button {{ font:inherit }} input,select {{ height:38px; border:1px solid #aeb5b8; background:var(--white); border-radius:2px; color:var(--ink); padding:0 10px }} input {{ width:240px }} label {{ color:var(--muted); font-size:12px }} button {{ border:1px solid #aeb5b8; background:var(--white); color:var(--ink); padding:8px 11px; cursor:pointer }} button:hover {{ border-color:var(--ink) }} button:focus-visible,[role=button]:focus-visible {{ outline:3px solid #91b8dc; outline-offset:2px }} .demo-toggle.active {{ background:var(--ink); border-color:var(--ink); color:white }} .clear-shortlist {{ color:var(--muted) }} .layer-note {{ min-height:24px; margin:10px 0 10px; color:var(--muted); font-size:12px }}
.demo-rail {{ min-height:46px; display:flex; align-items:center; gap:8px; margin:0 0 18px; border-block:1px solid var(--rule); padding:8px 0; overflow-x:auto }} .demo-rail-label {{ flex:0 0 auto; margin-right:5px; color:var(--muted); font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase }} .demo-chip {{ flex:0 0 auto; padding:5px 8px; border-color:var(--rule); font-size:11px }} .demo-empty {{ color:var(--muted); font-size:12px }}
.episodes {{ display:grid; grid-template-columns:1fr 1fr; gap:18px }} .episode {{ background:var(--white); border:1px solid var(--rule); padding:20px }} .episode-head {{ display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid var(--rule); padding-bottom:13px }} .kicker {{ margin:0 0 2px; font-weight:650; text-transform:capitalize }} .score {{ text-align:right }} .score b {{ display:block; font-size:21px }} .score span {{ color:var(--muted); font-size:11px }} .chart-heading {{ display:flex; justify-content:space-between; margin:17px 0 3px; font-size:12px; font-weight:650 }} .chart-heading span:last-child {{ color:var(--muted); font-weight:400 }} .chart-wrap {{ position:relative; padding-left:23px }} .progress-chart {{ display:block; width:100%; height:92px; overflow:visible }} .progress-chart polyline {{ fill:none; stroke:var(--teal); stroke-width:2; vector-effect:non-scaling-stroke }} .chart-grid {{ stroke:#e4e5e2; stroke-width:.6; vector-effect:non-scaling-stroke }} .y {{ position:absolute; left:0; color:#899196; font-size:9px }} .y100 {{ top:2px }} .y50 {{ top:42px }} .y0 {{ bottom:1px }}
.episode-actions {{ display:flex; align-items:flex-start; gap:13px }} .shortlist {{ padding:5px 8px; color:var(--blue); border-color:#9eb8cf; font-size:11px; white-space:nowrap }} .shortlist[aria-pressed=true] {{ color:white; background:var(--blue); border-color:var(--blue) }} .episode.shortlisted {{ border-top:3px solid var(--blue); padding-top:18px }}
.track-row {{ display:grid; grid-template-columns:88px 1fr; gap:8px; align-items:center; margin-top:8px; color:var(--muted); font-size:10px }} .event-track {{ position:relative; height:8px; background:#ecece8 }} .event-mark,.manual-mark {{ position:absolute; inset-block:0; min-width:2px; background:var(--gray); cursor:pointer }} .event-mark:hover,.manual-mark:hover {{ transform:scaleY(1.8); transform-origin:center }} .event-mark.learned-v2 {{ background:var(--blue) }} .event-mark.hybrid {{ background:var(--orange) }} .event-mark.aux {{ background:var(--rose) }} .event-mark.hand {{ background:var(--violet) }} .event-mark.learned {{ background:var(--teal) }} .manual-mark {{ background:#d8247c }} .human-row {{ margin-top:5px }} .axis {{ display:flex; justify-content:space-between; margin:3px 0 13px; padding-left:96px; color:var(--muted); font-size:10px }} .episode footer {{ display:flex; align-items:center; gap:14px; border-top:1px solid var(--rule); padding-top:11px; color:var(--muted); font-size:11px }} .watch {{ margin-left:auto; border:0; padding:0; color:var(--blue); background:transparent; font-size:11px; text-decoration:underline; text-underline-offset:3px }} details {{ margin-top:14px; border-top:1px solid var(--rule); padding-top:12px }} summary {{ cursor:pointer; font-weight:650 }} .detail-grid {{ display:grid; grid-template-columns:1.4fr 1fr; gap:18px; margin-top:12px }} table {{ width:100%; border-collapse:collapse; font-size:11px }} th,td {{ padding:6px 4px; text-align:left; border-bottom:1px solid #e6e7e5 }} th {{ color:var(--muted); font-weight:550 }} .time-link {{ border:0; background:transparent; padding:0; color:var(--blue); font-size:11px; text-decoration:underline; text-underline-offset:2px }} h4 {{ margin:0 0 9px }} .clean-list {{ display:flex; flex-wrap:wrap; gap:6px }} .clean-segment {{ border:1px solid #bdd8d1; background:#f0f8f5; padding:5px 7px; font-size:11px }} .muted {{ color:var(--muted) }}
.viewer {{ width:min(1040px,calc(100vw - 36px)); max-height:calc(100vh - 36px); border:1px solid #2d3439; padding:0; background:#111; color:white }} .viewer::backdrop {{ background:rgba(13,18,22,.78) }} .viewer-head {{ display:flex; justify-content:space-between; gap:18px; align-items:center; padding:14px 17px; background:#f5f4ef; color:var(--ink) }} .viewer-head p {{ margin:0; color:var(--muted); font-size:12px }} .viewer-close {{ padding:5px 10px; background:transparent }} .viewer video {{ display:block; width:100%; max-height:calc(100vh - 125px); background:#000 }}
.notes {{ margin-top:50px; border-top:1px solid var(--ink); padding-top:22px; display:grid; grid-template-columns:1fr 1fr; gap:34px }} .notes p {{ color:var(--muted) }} pre {{ max-height:220px; overflow:auto; background:#ecece7; padding:12px; font-size:10px; white-space:pre-wrap }} [hidden] {{ display:none!important }}
.event-mark.learned-v2 {{ background:var(--alert)!important }}
@media(max-width:900px) {{ main {{ padding:20px }} .intro,.hero,.reading,.notes {{ grid-template-columns:1fr }} .stats {{ grid-template-columns:1fr 1fr }} .stat {{ border-bottom:1px solid var(--rule) }} .flow {{ grid-template-columns:1fr }} .flow-arrow {{ transform:rotate(90deg); min-height:28px }} .episodes {{ grid-template-columns:1fr }} .catalog-head {{ align-items:start; flex-direction:column }} .controls {{ flex-wrap:wrap }} .episode-actions {{ flex-direction:column-reverse; align-items:flex-end }} }}
</style></head><body><main>
<div class="topline"><span class="brand">EGOFLOW</span><span class="version">Learned progress + learned interaction dynamics</span></div>
<section class="intro"><div><p class="overline">Robotics data curation</p><h1>Find the moments worth reviewing in long demonstrations.</h1><p>EgoFlow keeps the continuous progress estimate separate from behavioral claims, then adds sparse, source-attributed interaction deviations for a human curator.</p></div><div class="claim"><b>25 minutes → a searchable review index</b><span>18 complete episodes · whole-episode train/validation/test split</span></div></section>
<section class="stats"><div class="stat"><b>{len(episodes)}</b><span>episodes</span></div><div class="stat"><b>{total_seconds/60:.1f} min</b><span>indexed video</span></div><div class="stat"><b>{event_total}</b><span>learned review windows</span></div><div class="stat"><b>{manual_total}</b><span>reviewed spans</span></div><div class="stat"><b>{clean_total/60:.1f} min</b><span>candidate-free spans ≥10s</span></div></section>
<section class="hero"><div class="hero-media">{hero_media}</div><aside class="hero-copy"><p class="overline">Validation example · organizing plushies</p><h2>A subtle action switch, not a long stall.</h2><p>The reviewed span is 8–13 seconds. The learned expected-dynamics model identifies an unusual two-hand transition inside it.</p><div class="result"><b>{learned_event.get('start_sec','—')}–{learned_event.get('end_sec','—')} sec</b><span>{learned_event.get('surprise_mad','—')} MAD above expected hand dynamics · confidence {learned_event.get('confidence','—')}</span></div><p>The signal measures an unexpected interaction transition; the final semantic judgment stays with the reviewer.</p><div class="legend"><span class="key learned-v2">Learned interaction</span><span class="key learned">Learned progress</span><span class="key human">Reviewed span</span></div></aside></section>
<section class="method"><p class="overline">System, briefly</p><h2>Two learned signals; one review queue.</h2><p class="section-intro">The coarse model answers “how far through the demonstration are we?” The short-horizon model asks “does this hand transition look unlike the manipulation dynamics learned from training episodes?”</p><div class="flow"><div class="flow-node"><b>1 · Public RGB video</b><span>Frames only. Dense semantic annotations are used when available, but were absent in this public run.</span></div><div class="flow-arrow">→</div><div class="flow-node branch"><div><b>Frozen DINO features</b><span>2-layer BiGRU learns coarse progress.</span></div><div><b>2D hand landmarks</b><span>Small GRU learns expected next-hand state.</span></div></div><div class="flow-arrow">→</div><div class="flow-node"><b>2 · Source-attributed signals</b><span>Progress, centered slowdown deviation, learned interaction surprise, and disclosed auxiliary evidence.</span></div><div class="flow-arrow">→</div><div class="flow-node"><b>3 · Curate</b><span>Review suspicious windows or export candidate-free segments longer than ten seconds.</span></div></div></section>
<section class="reading"><div class="reading-panel"><p class="overline">Graph 1</p><h3>Learned task progress</h3><svg class="mini-chart" viewBox="0 0 420 120"><line x1="12" y1="104" x2="408" y2="104" class="base"/><path d="M12 102 C80 101 105 89 160 74 S245 48 300 29 S365 15 408 10" class="curve"/></svg><div class="mini-label"><span>start · 0%</span><span>later visual states score higher</span><span>end · 100%</span></div><p class="muted">This is the learned continuous output. It is not a “productive versus stalled” classifier.</p></div><div class="reading-panel"><p class="overline">Graph 2</p><h3>Slowdown deviation</h3><svg class="mini-chart" viewBox="0 0 420 120"><line x1="12" y1="65" x2="408" y2="65" class="base"/><path d="M12 65 L90 63 L120 66 L150 64 L185 26 L205 64 L260 67 L300 44 L320 65 L408 64" class="curve"/><line x1="185" y1="65" x2="185" y2="26" class="spike"/></svg><div class="mini-label"><span>faster than expected ↓</span><span>expected = 0</span><span>slower / review ↑</span></div><p class="muted">The sign is intentionally flipped for presentation: upward means expected rate minus actual rate, so suspicious slowdowns read as peaks.</p></div></section>
<section><div class="catalog-head"><div><p class="overline">Featured episodes</p><h2>Learned interaction signals across demonstrations.</h2></div><div class="controls"><input id="search" data-testid="episode-search" aria-label="Search episodes" placeholder="Search task or episode ID"><label for="layer">Show evidence</label><select id="layer" data-testid="evidence-filter">{evidence_options}</select></div></div><p class="layer-note" id="layer-note">{initial_layer_note}</p><div class="episodes" id="episodes">{''.join(cards)}</div></section>
<section class="notes"><div><p class="overline">Claim boundary</p><h2>Deliberately narrow.</h2><p>Learned v2 detects interaction surprise from 2D hand dynamics. It does not identify object A versus B, prove hesitation, or convert every slowdown into a behavioral label. All frozen baseline, derived, auxiliary, experimental, and human sources remain separately queryable.</p></div><div><details><summary>Frozen evaluation record</summary><pre>{metric_summary}</pre></details></div></section>
</main><dialog id="viewer" class="viewer"><div class="viewer-head"><div><strong id="viewer-title">Episode</strong><p id="viewer-meta">Starting at 0 seconds</p></div><button id="viewer-close" class="viewer-close" aria-label="Close video">Close</button></div><video id="viewer-video" controls playsinline preload="metadata"></video></dialog><script>
const cards=[...document.querySelectorAll('.episode')], search=document.querySelector('#search'), layer=document.querySelector('#layer'), note=document.querySelector('#layer-note');
const viewer=document.querySelector('#viewer'), viewerVideo=document.querySelector('#viewer-video'), viewerTitle=document.querySelector('#viewer-title'), viewerMeta=document.querySelector('#viewer-meta');
let stopAt=null;
const notes={{all:'All sparse candidate types are visible.',demo:'Learned interaction candidates and reviewed spans.','learned-v2':'Only self-supervised hand-dynamics candidates are shown.',hybrid:'Progress plus visual-dynamics candidates.',aux:'Frozen visual-dynamics candidates.',hand:'Video-derived hand-geometry candidates.',learned:'Progress-derived candidates.',human:'Only reviewed spans are shown.'}};
function apply(){{const query=search.value.toLowerCase(), selected=layer.value; cards.forEach(card=>card.hidden=!card.dataset.search.includes(query)); document.querySelectorAll('[data-source]').forEach(mark=>{{const source=mark.dataset.source; mark.hidden=selected==='all'?false:selected==='demo'?!['learned-v2','human'].includes(source):source!==selected}}); note.textContent=notes[selected]}}
function openViewer(button){{const start=Number(button.dataset.start||0), end=Number(button.dataset.end||0); stopAt=end>start?end:null; viewerTitle.textContent=button.dataset.title||'Episode'; viewerMeta.textContent=stopAt?`Review window · ${{start.toFixed(2)}}–${{end.toFixed(2)}} seconds`:`Full episode · starting at ${{start.toFixed(2)}} seconds`; viewerVideo.src=button.dataset.video; viewerVideo.addEventListener('loadedmetadata',()=>{{viewerVideo.currentTime=Math.min(start,Math.max(0,viewerVideo.duration-.05)); viewerVideo.play().catch(()=>{{}})}},{{once:true}}); if(!viewer.open)viewer.showModal()}}
document.addEventListener('click',event=>{{const action=event.target.closest('[data-action]'); if(action?.dataset.action==='watch') openViewer(action)}});
document.addEventListener('keydown',event=>{{if((event.key==='Enter'||event.key===' ')&&event.target.matches('[role=button][data-action=watch]')){{event.preventDefault(); openViewer(event.target)}}}});
viewerVideo.addEventListener('timeupdate',()=>{{if(stopAt&&viewerVideo.currentTime>=stopAt){{viewerVideo.pause(); stopAt=null}}}}); document.querySelector('#viewer-close').addEventListener('click',()=>viewer.close()); viewer.addEventListener('close',()=>{{viewerVideo.pause(); viewerVideo.removeAttribute('src'); viewerVideo.load(); stopAt=null}}); viewer.addEventListener('click',event=>{{if(event.target===viewer)viewer.close()}});
search.addEventListener('input',apply); layer.addEventListener('change',apply); apply();
</script></body></html>""", encoding="utf-8")
    return output
