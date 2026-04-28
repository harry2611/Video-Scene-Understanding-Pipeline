from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LabelScore(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class FrameMetadata(BaseModel):
    frame_index: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)
    path: str
    width: int | None = None
    height: int | None = None


class TemporalWindow(BaseModel):
    window_index: int = Field(ge=0)
    start_timestamp: float = Field(ge=0.0)
    end_timestamp: float = Field(ge=0.0)
    frame_indices: list[int]


class PipelineConfig(BaseModel):
    fps: float
    scene_detector: str
    classifier: str
    temporal_window_size: int
    temporal_stride: int
    clip_enabled: bool


class BenchmarkStage(BaseModel):
    name: str
    latency_ms: float = Field(ge=0.0)
    details: dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetrics(BaseModel):
    total_latency_ms: float = Field(ge=0.0)
    frames_per_second: float = Field(ge=0.0)
    total_frames: int = Field(ge=0)
    stage_latencies: list[BenchmarkStage]


class SceneMetadata(BaseModel):
    scene_id: str
    scene_index: int = Field(ge=0)
    start_timestamp: float = Field(ge=0.0)
    end_timestamp: float = Field(ge=0.0)
    duration: float = Field(ge=0.0)
    frame_indices: list[int]
    representative_frame: str | None = None
    labels: list[LabelScore] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    clip_embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoMetadata(BaseModel):
    model_config = ConfigDict(json_encoders={Path: str})

    video_id: str
    source: str
    local_video_path: str
    duration: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config: PipelineConfig
    frames: list[FrameMetadata]
    temporal_windows: list[TemporalWindow]
    scenes: list[SceneMetadata]
    benchmark: BenchmarkMetrics
    metadata_path: str | None = None


class SceneSearchResult(BaseModel):
    job_id: str
    scene_id: str
    scene_index: int
    start_timestamp: float
    end_timestamp: float
    score: float
    labels: list[LabelScore]
    representative_frame: str | None = None

