from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
        protected_namespaces=(),
    )

    database_url: str = Field(
        "postgresql+psycopg://scene:scene@localhost:55432/scene_pipeline",
        validation_alias=AliasChoices("DATABASE_URL", "PIPELINE_DATABASE_URL"),
    )
    redis_url: str = Field(
        "redis://localhost:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "PIPELINE_REDIS_URL"),
    )
    artifact_dir: Path = Field(
        Path("artifacts"),
        validation_alias=AliasChoices("PIPELINE_ARTIFACT_DIR", "ARTIFACT_DIR"),
    )
    default_fps: float = Field(1.0, validation_alias="PIPELINE_DEFAULT_FPS")
    model_name: str = Field("resnet50", validation_alias="PIPELINE_MODEL_NAME")
    scene_detector: str = Field("histogram", validation_alias="PIPELINE_SCENE_DETECTOR")
    histogram_threshold: float = Field(
        0.45,
        validation_alias="PIPELINE_HISTOGRAM_THRESHOLD",
    )
    temporal_window_size: int = Field(8, validation_alias="PIPELINE_TEMPORAL_WINDOW_SIZE")
    temporal_stride: int = Field(4, validation_alias="PIPELINE_TEMPORAL_STRIDE")
    device: str = Field("cpu", validation_alias="PIPELINE_DEVICE")
    classifier_batch_size: int = Field(8, validation_alias="PIPELINE_CLASSIFIER_BATCH_SIZE")
    top_k: int = Field(5, validation_alias="PIPELINE_TOP_K")
    enable_clip: bool = Field(False, validation_alias="PIPELINE_ENABLE_CLIP")
    clip_model_name: str = Field(
        "openai/clip-vit-base-patch32",
        validation_alias="PIPELINE_CLIP_MODEL_NAME",
    )
    enable_quality_scoring: bool = Field(True, validation_alias="PIPELINE_ENABLE_QUALITY_SCORING")
    blur_threshold: float = Field(100.0, validation_alias="PIPELINE_BLUR_THRESHOLD")
    brightness_min: float = Field(35.0, validation_alias="PIPELINE_BRIGHTNESS_MIN")
    brightness_max: float = Field(220.0, validation_alias="PIPELINE_BRIGHTNESS_MAX")
    enable_multi_gpu: bool = Field(False, validation_alias="PIPELINE_ENABLE_MULTI_GPU")
    queue_name: str = Field("video-scene-jobs", validation_alias="PIPELINE_QUEUE_NAME")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    return settings
