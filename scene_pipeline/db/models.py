from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from scene_pipeline.db.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class VideoJob(Base):
    __tablename__ = "video_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    metadata_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    benchmark_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    scenes: Mapped[list["SceneRecord"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="SceneRecord.scene_index",
    )


class SceneRecord(Base):
    __tablename__ = "scene_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("video_jobs.job_id", ondelete="CASCADE"))
    scene_id: Mapped[str] = mapped_column(String(128), index=True)
    scene_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    end_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    representative_frame: Mapped[str | None] = mapped_column(Text, nullable=True)
    labels: Mapped[list | None] = mapped_column(JSON, nullable=True)
    clip_embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    extra_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    job: Mapped[VideoJob] = relationship(back_populates="scenes")

