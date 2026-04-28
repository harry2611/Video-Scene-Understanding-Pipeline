from __future__ import annotations

from pydantic import BaseModel, Field


class VideoRequest(BaseModel):
    source: str = Field(..., description="Local MP4 path, direct URL, or YouTube URL")
    fps: float | None = Field(default=None, gt=0)
    scene_detector: str | None = Field(default=None, pattern="^(histogram|pyscenedetect)$")
    classifier: str | None = Field(default=None, pattern="^(resnet50|efficientnet_b0)$")
    temporal_window_size: int | None = Field(default=None, gt=0)
    temporal_stride: int | None = Field(default=None, gt=0)
    enable_clip: bool | None = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    queue_job_id: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    source: str
    status: str
    error: str | None = None
    benchmark: dict | None = None

