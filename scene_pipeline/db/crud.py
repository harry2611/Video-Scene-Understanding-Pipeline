from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from scene_pipeline.db.models import SceneRecord, VideoJob
from scene_pipeline.schemas import VideoMetadata


def create_job(session: Session, job_id: str, source: str, status: str = "queued") -> VideoJob:
    job = VideoJob(job_id=job_id, source=source, status=status, updated_at=_now())
    session.merge(job)
    session.commit()
    return job


def update_job_status(
    session: Session,
    job_id: str,
    status: str,
    error: str | None = None,
) -> None:
    job = session.get(VideoJob, job_id)
    if job is None:
        job = VideoJob(job_id=job_id, source="", status=status)
        session.add(job)
    job.status = status
    job.error = error
    job.updated_at = _now()
    session.commit()


def save_metadata(session: Session, metadata: VideoMetadata, status: str = "completed") -> None:
    job = session.get(VideoJob, metadata.video_id)
    if job is None:
        job = VideoJob(job_id=metadata.video_id, source=metadata.source)
        session.add(job)
    job.source = metadata.source
    job.status = status
    job.metadata_payload = metadata.model_dump(mode="json")
    job.benchmark_payload = metadata.benchmark.model_dump(mode="json")
    job.error = None
    job.updated_at = _now()
    job.scenes.clear()

    for scene in metadata.scenes:
        job.scenes.append(
            SceneRecord(
                job_id=metadata.video_id,
                scene_id=scene.scene_id,
                scene_index=scene.scene_index,
                start_timestamp=scene.start_timestamp,
                end_timestamp=scene.end_timestamp,
                representative_frame=scene.representative_frame,
                labels=[label.model_dump(mode="json") for label in scene.labels],
                clip_embedding=scene.clip_embedding,
                quality_score=scene.quality.data_quality_score if scene.quality else None,
                quality_payload=scene.quality.model_dump(mode="json") if scene.quality else None,
                extra_metadata=scene.metadata,
            )
        )
    session.commit()


def get_job(session: Session, job_id: str) -> VideoJob | None:
    return session.get(VideoJob, job_id)


def get_metadata(session: Session, job_id: str) -> dict | None:
    job = session.get(VideoJob, job_id)
    return job.metadata_payload if job else None


def get_scene_embeddings(session: Session, job_id: str | None = None) -> list[SceneRecord]:
    statement = select(SceneRecord).where(SceneRecord.clip_embedding.is_not(None))
    if job_id:
        statement = statement.where(SceneRecord.job_id == job_id)
    return list(session.scalars(statement).all())


def aggregate_metrics(session: Session) -> dict:
    total_jobs = session.scalar(select(func.count(VideoJob.job_id))) or 0
    completed_jobs = (
        session.scalar(select(func.count(VideoJob.job_id)).where(VideoJob.status == "completed")) or 0
    )
    failed_jobs = (
        session.scalar(select(func.count(VideoJob.job_id)).where(VideoJob.status == "failed")) or 0
    )
    total_scenes = session.scalar(select(func.count(SceneRecord.id))) or 0
    jobs = session.scalars(select(VideoJob).where(VideoJob.benchmark_payload.is_not(None))).all()
    fps_values = [
        float(job.benchmark_payload.get("frames_per_second", 0.0))
        for job in jobs
        if job.benchmark_payload
    ]
    latency_values = [
        float(job.benchmark_payload.get("total_latency_ms", 0.0))
        for job in jobs
        if job.benchmark_payload
    ]
    quality_values = [
        float(scene.quality_score)
        for scene in session.scalars(select(SceneRecord).where(SceneRecord.quality_score.is_not(None)))
        if scene.quality_score is not None
    ]
    low_quality_scenes = sum(1 for value in quality_values if value < 0.55)
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "total_scenes": total_scenes,
        "average_frames_per_second": _average(fps_values),
        "average_total_latency_ms": _average(latency_values),
        "average_quality_score": _average(quality_values),
        "low_quality_scenes": low_quality_scenes,
    }


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _now() -> datetime:
    return datetime.now(timezone.utc)
