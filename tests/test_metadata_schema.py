import json
from pathlib import Path

from jsonschema import Draft202012Validator

from scene_pipeline.schemas import (
    BenchmarkMetrics,
    FrameMetadata,
    PipelineConfig,
    SceneMetadata,
    VideoMetadata,
)


def test_video_metadata_matches_downstream_schema() -> None:
    schema = json.loads(Path("schemas/video_metadata.schema.json").read_text(encoding="utf-8"))
    metadata = VideoMetadata(
        video_id="test",
        source="sample.mp4",
        local_video_path="/tmp/sample.mp4",
        duration=1.0,
        config=PipelineConfig(
            fps=1.0,
            scene_detector="histogram",
            classifier="resnet50",
            temporal_window_size=8,
            temporal_stride=4,
            clip_enabled=False,
        ),
        frames=[FrameMetadata(frame_index=0, timestamp=0.0, path="/tmp/frame.jpg")],
        temporal_windows=[],
        scenes=[
            SceneMetadata(
                scene_id="test:scene_0000",
                scene_index=0,
                start_timestamp=0.0,
                end_timestamp=0.0,
                duration=0.0,
                frame_indices=[0],
                labels=[],
                confidence=0.0,
            )
        ],
        benchmark=BenchmarkMetrics(
            total_latency_ms=1.0,
            frames_per_second=1.0,
            total_frames=1,
            stage_latencies=[],
        ),
    )

    Draft202012Validator(schema).validate(metadata.model_dump(mode="json"))

