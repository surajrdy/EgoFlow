"""Bounded Modal extraction for EgoFlow v2 video-derived hand geometry."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

import modal


app = modal.App("egoflow-hand-v2")
data_volume = modal.Volume.from_name("egoverse-data", create_if_missing=True)
hand_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libegl1", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy>=1.26,<2",
        "mediapipe==0.10.21",
        "opencv-python-headless==4.11.0.86",
    )
    .env({"PYTHONPATH": "/opt"})
    .add_local_dir("hackathon", remote_path="/opt/hackathon", copy=True)
)


@app.function(
    image=hand_image,
    cpu=4,
    memory=8192,
    timeout=10 * 60,
    max_containers=8,
    volumes={"/vol": data_volume},
)
def extract_hand_geometry(episode_id: str, sample_fps: float = 8.0) -> dict[str, object]:
    from hackathon.egoflow.hand_interaction import (
        detect_aborted_reaches,
        extract_hand_observations,
    )

    if len(episode_id) != 24 or any(character not in "0123456789abcdef" for character in episode_id.lower()):
        raise ValueError("episode_id must be a 24-character hexadecimal Mecka ID")
    with tempfile.TemporaryDirectory(prefix="egoflow-hand-") as directory:
        video_path = Path(directory) / "source.mp4"
        url = f"https://partners.mecka.ai/api/egoverse/uploads/{episode_id}/video?redirect=1"
        with urlopen(url, timeout=120) as response, video_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        extraction = extract_hand_observations(video_path, sample_fps=sample_fps)
    candidates = detect_aborted_reaches(
        extraction["observations"],
        sample_fps=float(extraction["sample_fps"]),
    )
    result = {
        "episode_id": episode_id,
        "schema": "egoflow.hand_geometry.v1",
        "video_derived_only": True,
        **extraction,
        "candidates": candidates,
    }
    output_path = Path("/vol/egoflow/hand_v2") / f"{episode_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    data_volume.commit()
    return {
        "episode_id": episode_id,
        "output_path": str(output_path),
        "sampled_frames": extraction["sampled_frames"],
        "detected_frames": extraction["detected_frames"],
        "detection_rate": extraction["detection_rate"],
        "candidates": candidates,
    }


@app.local_entrypoint()
def main(episode_ids: str = "69bb1239efeadec2abedad96", sample_fps: float = 8.0) -> None:
    ids = [value.strip() for value in episode_ids.split(",") if value.strip()]
    if not 1 <= len(ids) <= 80:
        raise ValueError("episode_ids must contain 1-80 comma-separated IDs")
    results = list(extract_hand_geometry.map(ids, kwargs={"sample_fps": sample_fps}))
    print(json.dumps({"event": "hand_v2_complete", "results": results}))
