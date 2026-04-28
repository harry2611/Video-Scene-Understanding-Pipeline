from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from scene_pipeline.api.models import JobResponse, JobStatusResponse, VideoRequest
from scene_pipeline.config import Settings, get_settings
from scene_pipeline.core.embeddings import CLIPSceneEmbedder, cosine_similarity
from scene_pipeline.core.pipeline import PipelineOptions, VideoScenePipeline
from scene_pipeline.db.crud import (
    aggregate_metrics,
    create_job,
    get_job,
    get_metadata,
    get_scene_embeddings,
    save_metadata,
    update_job_status,
)
from scene_pipeline.db.database import get_session, init_db
from scene_pipeline.schemas import SceneSearchResult, VideoMetadata
from scene_pipeline.worker.jobs import process_video_job
from scene_pipeline.worker.queue import enqueue_video_job


app = FastAPI(title="Video Scene Understanding Pipeline", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
settings.artifact_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/artifacts",
    StaticFiles(directory=str(settings.artifact_dir)),
    name="artifacts",
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/videos", response_model=JobResponse)
def submit_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> JobResponse:
    job_id = str(uuid.uuid4())
    options = _options_from_request(request, settings)
    create_job(session, job_id=job_id, source=request.source, status="queued")
    try:
        queue_job_id = enqueue_video_job(job_id, request.source, options.__dict__)
        return JobResponse(job_id=job_id, status="queued", queue_job_id=queue_job_id)
    except Exception:
        background_tasks.add_task(process_video_job, job_id, request.source, options.__dict__)
        return JobResponse(job_id=job_id, status="queued", queue_job_id=None)


@app.post("/videos/sync", response_model=VideoMetadata)
def process_video_sync(
    request: VideoRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> VideoMetadata:
    job_id = str(uuid.uuid4())
    options = _options_from_request(request, settings)
    create_job(session, job_id=job_id, source=request.source, status="processing")
    try:
        metadata = VideoScenePipeline(settings).process(request.source, job_id=job_id, options=options)
        save_metadata(session, metadata, status="completed")
        return metadata
    except Exception as exc:
        update_job_status(session, job_id, "failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, session: Session = Depends(get_session)) -> JobStatusResponse:
    job = get_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        source=job.source,
        status=job.status,
        error=job.error,
        benchmark=job.benchmark_payload,
    )


@app.get("/videos/{job_id}/metadata")
def video_metadata(job_id: str, session: Session = Depends(get_session)) -> dict:
    metadata = get_metadata(session, job_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Metadata not found")
    return metadata


@app.get("/search", response_model=list[SceneSearchResult])
def search_scenes(
    query: str = Query(..., min_length=1),
    job_id: str | None = None,
    top_k: int = Query(10, gt=0, le=100),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> list[SceneSearchResult]:
    scenes = get_scene_embeddings(session, job_id=job_id)
    if not scenes:
        return []
    try:
        text_embedding = CLIPSceneEmbedder(settings.clip_model_name, settings.device).embed_text(query)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"CLIP search unavailable: {exc}") from exc

    ranked = sorted(
        (
            (scene, cosine_similarity(text_embedding, scene.clip_embedding or []))
            for scene in scenes
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    return [
        SceneSearchResult(
            job_id=scene.job_id,
            scene_id=scene.scene_id,
            scene_index=scene.scene_index,
            start_timestamp=scene.start_timestamp,
            end_timestamp=scene.end_timestamp,
            score=float(score),
            labels=scene.labels or [],
            representative_frame=_artifact_url(scene.representative_frame, settings),
        )
        for scene, score in ranked
    ]


@app.get("/metrics")
def metrics(session: Session = Depends(get_session)) -> dict:
    return aggregate_metrics(session)


def _options_from_request(request: VideoRequest, settings: Settings) -> PipelineOptions:
    return PipelineOptions(
        fps=request.fps if request.fps is not None else settings.default_fps,
        scene_detector=request.scene_detector or settings.scene_detector,
        classifier=request.classifier or settings.model_name,
        temporal_window_size=request.temporal_window_size or settings.temporal_window_size,
        temporal_stride=request.temporal_stride or settings.temporal_stride,
        enable_clip=settings.enable_clip if request.enable_clip is None else request.enable_clip,
    )


def _artifact_url(path: str | None, settings: Settings) -> str | None:
    if not path:
        return None
    try:
        relative = Path(path).resolve().relative_to(settings.artifact_dir.resolve())
    except ValueError:
        return path
    return f"/artifacts/{relative.as_posix()}"
