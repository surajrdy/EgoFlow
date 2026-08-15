"""Dense language-span parsing and deterministic frame assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class AnnotationSpan:
    text: str
    start_frame: int
    end_frame: int
    source_index: int = -1

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError(
                f"invalid annotation bounds {self.start_frame}:{self.end_frame}"
            )


@dataclass(frozen=True)
class FrameAnnotation:
    primary_span_id: int
    primary_text: str
    active_span_ids: tuple[int, ...]
    active_texts: tuple[str, ...]
    inherited: bool


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    normalized = {str(k).lower(): v for k, v in record.items()}
    for name in names:
        if name in normalized:
            return normalized[name]
    return None


def spans_from_records(records: Iterable[Mapping[str, Any]]) -> list[AnnotationSpan]:
    """Normalize common EgoVerse/JSON annotation field names.

    End bounds are treated as inclusive, matching the visualization scripts used by
    many robotics Zarr datasets. A caller can subtract one before constructing a
    record if its source uses exclusive bounds.
    """

    spans: list[AnnotationSpan] = []
    for index, record in enumerate(records):
        text = _first(record, ("text", "annotation", "language", "label", "description"))
        start = _first(
            record,
            ("start_frame", "start_idx", "start_index", "start", "frame_start"),
        )
        end = _first(
            record, ("end_frame", "end_idx", "end_index", "end", "frame_end")
        )
        if text is None or start is None or end is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        spans.append(AnnotationSpan(text, int(start), int(end), index))
    return sorted(spans, key=lambda span: (span.start_frame, span.end_frame, span.source_index))


def assign_annotations(
    frame_indices: Sequence[int],
    spans: Sequence[AnnotationSpan],
    *,
    task_description: str,
    max_gap_frames: int,
) -> list[FrameAnnotation]:
    """Assign language context using latest-started overlap and short-gap inheritance."""

    if max_gap_frames < 0:
        raise ValueError("max_gap_frames cannot be negative")
    ordered = list(spans)
    assigned: list[FrameAnnotation] = []
    for frame in frame_indices:
        active = [
            (span_id, span)
            for span_id, span in enumerate(ordered)
            if span.start_frame <= int(frame) <= span.end_frame
        ]
        inherited = False
        if active:
            # Later start wins; ties are broken by later source order.
            primary_id, primary = max(
                active, key=lambda pair: (pair[1].start_frame, pair[1].source_index)
            )
            active = sorted(active, key=lambda pair: (pair[1].start_frame, pair[0]))
        else:
            previous = [
                (span_id, span)
                for span_id, span in enumerate(ordered)
                if span.end_frame < int(frame)
                and int(frame) - span.end_frame <= max_gap_frames
            ]
            if previous:
                primary_id, primary = max(
                    previous, key=lambda pair: (pair[1].end_frame, pair[1].start_frame)
                )
                active = []
                inherited = True
            else:
                primary_id, primary = -1, None
        assigned.append(
            FrameAnnotation(
                primary_span_id=primary_id,
                primary_text=primary.text if primary is not None else task_description,
                active_span_ids=tuple(pair[0] for pair in active),
                active_texts=tuple(pair[1].text for pair in active),
                inherited=inherited,
            )
        )
    return assigned
