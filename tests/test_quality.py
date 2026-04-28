from scene_pipeline.core.quality import choose_representative_frame, score_scene_quality
from scene_pipeline.schemas import FrameMetadata, FrameQuality


def test_scene_quality_penalizes_low_quality_ratio() -> None:
    frames = [
        FrameMetadata(
            frame_index=0,
            timestamp=0.0,
            path="a.jpg",
            quality=FrameQuality(
                blur_variance=20.0,
                brightness=20.0,
                quality_score=0.25,
                is_low_quality=True,
                flags=["blurry", "too_dark"],
            ),
        ),
        FrameMetadata(
            frame_index=1,
            timestamp=1.0,
            path="b.jpg",
            quality=FrameQuality(
                blur_variance=240.0,
                brightness=120.0,
                quality_score=0.95,
                is_low_quality=False,
                flags=[],
            ),
        ),
    ]

    quality = score_scene_quality(frames)

    assert quality.low_quality_frame_ratio == 0.5
    assert quality.recommended_action in {"review", "reject"}
    assert "high_low_quality_ratio" in quality.flags


def test_representative_frame_prefers_high_quality_frame() -> None:
    frames = [
        FrameMetadata(
            frame_index=0,
            timestamp=0.0,
            path="bad.jpg",
            quality=FrameQuality(
                blur_variance=10.0,
                brightness=10.0,
                quality_score=0.1,
                is_low_quality=True,
                flags=["blurry"],
            ),
        ),
        FrameMetadata(
            frame_index=1,
            timestamp=1.0,
            path="good.jpg",
            quality=FrameQuality(
                blur_variance=200.0,
                brightness=128.0,
                quality_score=0.98,
                is_low_quality=False,
                flags=[],
            ),
        ),
    ]

    assert choose_representative_frame(frames) == "good.jpg"

