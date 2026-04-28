from __future__ import annotations

from pathlib import Path

from scene_pipeline.schemas import FrameMetadata


def _histogram_signature(frame_path: str):
    import cv2

    image = cv2.imread(frame_path)
    if image is None:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def detect_histogram_scenes(
    frames: list[FrameMetadata],
    threshold: float = 0.45,
) -> list[tuple[int, int]]:
    """Detect scene segments as inclusive frame-index ranges."""

    if not frames:
        return []
    if len(frames) == 1:
        return [(0, 0)]

    import cv2

    scenes: list[tuple[int, int]] = []
    start = 0
    previous = _histogram_signature(frames[0].path)

    for index in range(1, len(frames)):
        current = _histogram_signature(frames[index].path)
        if previous is None or current is None:
            previous = current
            continue
        correlation = cv2.compareHist(previous, current, cv2.HISTCMP_CORREL)
        difference = 1.0 - max(min(correlation, 1.0), -1.0)
        if difference >= threshold:
            scenes.append((start, index - 1))
            start = index
        previous = current

    scenes.append((start, len(frames) - 1))
    return scenes


def _nearest_frame_index(frames: list[FrameMetadata], timestamp: float) -> int:
    return min(
        range(len(frames)),
        key=lambda index: abs(frames[index].timestamp - timestamp),
    )


def detect_pyscenedetect_scenes(
    video_path: Path,
    frames: list[FrameMetadata],
    threshold: float = 27.0,
) -> list[tuple[int, int]]:
    if not frames:
        return []

    try:
        from scenedetect import ContentDetector, detect
    except ImportError as exc:
        raise RuntimeError("PySceneDetect is not installed") from exc

    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
    if not scene_list:
        return [(0, len(frames) - 1)]

    ranges: list[tuple[int, int]] = []
    for start_time, end_time in scene_list:
        start_index = _nearest_frame_index(frames, start_time.get_seconds())
        end_index = _nearest_frame_index(frames, max(end_time.get_seconds(), 0.0))
        if end_index < start_index:
            end_index = start_index
        ranges.append((start_index, min(end_index, len(frames) - 1)))
    return ranges


def detect_scenes(
    video_path: Path,
    frames: list[FrameMetadata],
    method: str,
    histogram_threshold: float,
) -> list[tuple[int, int]]:
    method = method.lower()
    if method in {"pyscenedetect", "scene_detect", "content"}:
        try:
            return detect_pyscenedetect_scenes(video_path, frames)
        except RuntimeError:
            return detect_histogram_scenes(frames, threshold=histogram_threshold)
    if method != "histogram":
        raise ValueError(f"Unsupported scene detector: {method}")
    return detect_histogram_scenes(frames, threshold=histogram_threshold)
