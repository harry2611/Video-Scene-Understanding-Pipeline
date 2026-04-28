from __future__ import annotations

from scene_pipeline.schemas import FrameMetadata, TemporalWindow


def sliding_windows(
    frames: list[FrameMetadata],
    window_size: int,
    stride: int,
) -> list[TemporalWindow]:
    if not frames:
        return []
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    windows: list[TemporalWindow] = []
    window_index = 0
    last_start = max(len(frames) - window_size, 0)

    starts = list(range(0, len(frames), stride))
    if starts[-1] != last_start:
        starts.append(last_start)

    seen: set[int] = set()
    for start in starts:
        start = min(start, last_start)
        if start in seen:
            continue
        seen.add(start)
        batch = frames[start : start + window_size]
        windows.append(
            TemporalWindow(
                window_index=window_index,
                start_timestamp=batch[0].timestamp,
                end_timestamp=batch[-1].timestamp,
                frame_indices=[frame.frame_index for frame in batch],
            )
        )
        window_index += 1

    return windows

