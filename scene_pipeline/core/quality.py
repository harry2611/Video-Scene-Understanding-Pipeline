from __future__ import annotations

from statistics import mean, pstdev

from scene_pipeline.schemas import FrameMetadata, FrameQuality, SceneQuality


def annotate_frame_quality(
    frames: list[FrameMetadata],
    blur_threshold: float,
    brightness_min: float,
    brightness_max: float,
) -> list[FrameMetadata]:
    for frame in frames:
        frame.quality = analyze_frame_quality(
            frame.path,
            blur_threshold=blur_threshold,
            brightness_min=brightness_min,
            brightness_max=brightness_max,
        )
    return frames


def analyze_frame_quality(
    frame_path: str,
    blur_threshold: float,
    brightness_min: float,
    brightness_max: float,
) -> FrameQuality:
    try:
        import cv2
    except ImportError:
        return FrameQuality(
            blur_variance=0.0,
            brightness=0.0,
            quality_score=0.0,
            is_low_quality=True,
            flags=["quality_backend_missing"],
        )

    image = cv2.imread(frame_path)
    if image is None:
        return FrameQuality(
            blur_variance=0.0,
            brightness=0.0,
            quality_score=0.0,
            is_low_quality=True,
            flags=["unreadable_frame"],
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    flags: list[str] = []
    if blur_variance < blur_threshold:
        flags.append("blurry")
    if brightness < brightness_min:
        flags.append("too_dark")
    if brightness > brightness_max:
        flags.append("too_bright")

    blur_component = min(blur_variance / max(blur_threshold, 1e-9), 1.0)
    brightness_component = _brightness_component(
        brightness=brightness,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
    )
    quality_score = _clamp((0.58 * blur_component) + (0.42 * brightness_component))
    is_low_quality = bool(flags) or quality_score < 0.55

    return FrameQuality(
        blur_variance=blur_variance,
        brightness=brightness,
        quality_score=quality_score,
        is_low_quality=is_low_quality,
        flags=flags,
    )


def score_scene_quality(frames: list[FrameMetadata]) -> SceneQuality:
    qualities = [frame.quality for frame in frames if frame.quality is not None]
    total_count = len(frames)
    if not qualities:
        return SceneQuality(
            data_quality_score=0.0,
            low_quality_frame_ratio=1.0 if total_count else 0.0,
            usable_frame_count=0,
            total_frame_count=total_count,
            quality_grade="unknown",
            recommended_action="review",
            flags=["missing_frame_quality"],
        )

    frame_scores = [quality.quality_score for quality in qualities]
    low_quality_count = sum(1 for quality in qualities if quality.is_low_quality)
    usable_count = len(qualities) - low_quality_count
    low_quality_ratio = low_quality_count / len(qualities)
    score_stddev = pstdev(frame_scores) if len(frame_scores) > 1 else 0.0
    brightness_values = [quality.brightness for quality in qualities]
    brightness_stddev = pstdev(brightness_values) if len(brightness_values) > 1 else 0.0

    flags = sorted({flag for quality in qualities for flag in quality.flags})
    if low_quality_ratio >= 0.35:
        flags.append("high_low_quality_ratio")
    if score_stddev >= 0.25:
        flags.append("unstable_quality")
    if brightness_stddev >= 45.0:
        flags.append("brightness_flicker")

    base_score = mean(frame_scores)
    temporal_penalty = min(score_stddev * 0.45, 0.18)
    low_quality_penalty = low_quality_ratio * 0.32
    data_quality_score = _clamp(base_score - temporal_penalty - low_quality_penalty)

    return SceneQuality(
        data_quality_score=data_quality_score,
        low_quality_frame_ratio=low_quality_ratio,
        usable_frame_count=usable_count,
        total_frame_count=total_count,
        quality_grade=_quality_grade(data_quality_score),
        recommended_action=_recommended_action(data_quality_score, low_quality_ratio),
        flags=flags,
    )


def choose_representative_frame(frames: list[FrameMetadata]) -> str | None:
    if not frames:
        return None
    if any(frame.quality is not None for frame in frames):
        best_frame = max(
            frames,
            key=lambda frame: frame.quality.quality_score if frame.quality else 0.0,
        )
        return best_frame.path
    return frames[len(frames) // 2].path


def _brightness_component(
    brightness: float,
    brightness_min: float,
    brightness_max: float,
) -> float:
    if brightness < brightness_min:
        return _clamp(brightness / max(brightness_min, 1e-9))
    if brightness > brightness_max:
        return _clamp((255.0 - brightness) / max(255.0 - brightness_max, 1e-9))
    target = (brightness_min + brightness_max) / 2.0
    half_range = max((brightness_max - brightness_min) / 2.0, 1e-9)
    distance = abs(brightness - target) / half_range
    return _clamp(1.0 - (distance * 0.18))


def _quality_grade(score: float) -> str:
    if score >= 0.88:
        return "excellent"
    if score >= 0.72:
        return "good"
    if score >= 0.55:
        return "review"
    return "poor"


def _recommended_action(score: float, low_quality_ratio: float) -> str:
    if score >= 0.72 and low_quality_ratio < 0.25:
        return "accept"
    if score >= 0.55 and low_quality_ratio < 0.5:
        return "review"
    return "reject"


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))

