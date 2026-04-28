from __future__ import annotations

from rq import Worker

from scene_pipeline.config import get_settings
from scene_pipeline.core.pipeline import PipelineOptions, VideoScenePipeline
from scene_pipeline.db.crud import save_metadata, update_job_status
from scene_pipeline.db.database import SessionLocal, init_db
from scene_pipeline.worker.queue import get_redis_connection


def process_video_job(job_id: str, source: str, options_payload: dict) -> dict:
    init_db()
    session = SessionLocal()
    try:
        update_job_status(session, job_id, "processing")
        settings = get_settings()
        pipeline = VideoScenePipeline(settings)
        options = PipelineOptions(**options_payload)
        metadata = pipeline.process(source=source, job_id=job_id, options=options)
        save_metadata(session, metadata, status="completed")
        return {
            "job_id": job_id,
            "status": "completed",
            "frames": len(metadata.frames),
            "scenes": len(metadata.scenes),
            "frames_per_second": metadata.benchmark.frames_per_second,
        }
    except Exception as exc:
        update_job_status(session, job_id, "failed", error=str(exc))
        raise
    finally:
        session.close()


def main() -> None:
    settings = get_settings()
    init_db()
    worker = Worker([settings.queue_name], connection=get_redis_connection())
    worker.work()


if __name__ == "__main__":
    main()

