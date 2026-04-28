from __future__ import annotations

from redis import Redis
from rq import Queue

from scene_pipeline.config import get_settings


def get_redis_connection() -> Redis:
    settings = get_settings()
    return Redis.from_url(settings.redis_url)


def get_queue() -> Queue:
    settings = get_settings()
    return Queue(settings.queue_name, connection=get_redis_connection())


def enqueue_video_job(job_id: str, source: str, options: dict) -> str:
    queue = get_queue()
    job = queue.enqueue(
        "scene_pipeline.worker.jobs.process_video_job",
        job_id,
        source,
        options,
        job_timeout="2h",
        result_ttl=86400,
    )
    return job.id

