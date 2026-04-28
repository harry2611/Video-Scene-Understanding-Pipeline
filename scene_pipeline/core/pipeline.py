from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from scene_pipeline.config import Settings, get_settings
from scene_pipeline.core.benchmark import BenchmarkRecorder
from scene_pipeline.core.classification import NullSceneClassifier, build_classifier
from scene_pipeline.core.embeddings import CLIPSceneEmbedder
from scene_pipeline.core.ingestion import extract_frames, probe_duration, resolve_video_source
from scene_pipeline.core.quality import (
    annotate_frame_quality,
    choose_representative_frame,
    score_scene_quality,
)
from scene_pipeline.core.scene_detection import detect_scenes
from scene_pipeline.core.temporal import sliding_windows
from scene_pipeline.schemas import PipelineConfig, SceneMetadata, VideoMetadata


@dataclass
class PipelineOptions:
    fps: float
    scene_detector: str
    classifier: str
    temporal_window_size: int
    temporal_stride: int
    enable_clip: bool
    enable_quality_scoring: bool = True
    enable_multi_gpu: bool = True

    @classmethod
    def from_settings(cls, settings: Settings) -> "PipelineOptions":
        return cls(
            fps=settings.default_fps,
            scene_detector=settings.scene_detector,
            classifier=settings.model_name,
            temporal_window_size=settings.temporal_window_size,
            temporal_stride=settings.temporal_stride,
            enable_clip=settings.enable_clip,
            enable_quality_scoring=settings.enable_quality_scoring,
            enable_multi_gpu=settings.enable_multi_gpu,
        )


class VideoScenePipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def process(
        self,
        source: str,
        job_id: str | None = None,
        options: PipelineOptions | None = None,
    ) -> VideoMetadata:
        job_id = job_id or str(uuid.uuid4())
        options = options or PipelineOptions.from_settings(self.settings)
        job_dir = self.settings.artifact_dir / job_id
        frame_dir = job_dir / "frames"
        recorder = BenchmarkRecorder()

        with recorder.stage("ingestion", source=source):
            local_video_path = resolve_video_source(source, job_dir)
            try:
                duration = probe_duration(local_video_path)
            except Exception:
                duration = None

        with recorder.stage("frame_extraction", fps=options.fps):
            frames = extract_frames(local_video_path, frame_dir, options.fps)
        if not frames:
            raise RuntimeError("FFmpeg extracted zero frames from the video")

        if options.enable_quality_scoring:
            with recorder.stage(
                "quality_scoring",
                blur_threshold=self.settings.blur_threshold,
                brightness_min=self.settings.brightness_min,
                brightness_max=self.settings.brightness_max,
            ):
                frames = annotate_frame_quality(
                    frames,
                    blur_threshold=self.settings.blur_threshold,
                    brightness_min=self.settings.brightness_min,
                    brightness_max=self.settings.brightness_max,
                )

        with recorder.stage("scene_detection", detector=options.scene_detector):
            scene_ranges = detect_scenes(
                local_video_path,
                frames,
                method=options.scene_detector,
                histogram_threshold=self.settings.histogram_threshold,
            )

        with recorder.stage(
            "temporal_batching",
            window_size=options.temporal_window_size,
            stride=options.temporal_stride,
        ):
            temporal_windows = sliding_windows(
                frames,
                window_size=options.temporal_window_size,
                stride=options.temporal_stride,
            )

        try:
            classifier = build_classifier(
                options.classifier,
                device=self.settings.device,
                top_k=self.settings.top_k,
                batch_size=self.settings.classifier_batch_size,
                use_data_parallel=options.enable_multi_gpu,
            )
        except Exception:
            classifier = NullSceneClassifier()

        scenes: list[SceneMetadata] = []
        with recorder.stage(
            "frame_classification",
            classifier=options.classifier,
            multi_gpu_enabled=options.enable_multi_gpu,
        ) as classification_details:
            for scene_index, (start_index, end_index) in enumerate(scene_ranges):
                scene_frames = frames[start_index : end_index + 1]
                scene_extra = {
                    "start_frame_index": start_index,
                    "end_frame_index": end_index,
                }
                try:
                    labels = classifier.classify_scene(scene_frames)
                except Exception as exc:
                    classifier = NullSceneClassifier()
                    labels = classifier.classify_scene(scene_frames)
                    scene_extra["classification_error"] = str(exc)
                representative = choose_representative_frame(scene_frames)
                confidence = labels[0].confidence if labels else 0.0
                quality = score_scene_quality(scene_frames) if options.enable_quality_scoring else None
                scenes.append(
                    SceneMetadata(
                        scene_id=f"{job_id}:scene_{scene_index:04d}",
                        scene_index=scene_index,
                        start_timestamp=scene_frames[0].timestamp,
                        end_timestamp=scene_frames[-1].timestamp,
                        duration=max(scene_frames[-1].timestamp - scene_frames[0].timestamp, 0.0),
                        frame_indices=[frame.frame_index for frame in scene_frames],
                        representative_frame=representative,
                        labels=labels,
                        confidence=confidence,
                        quality=quality,
                        metadata=scene_extra,
                    )
                )
            classification_details.update(classifier.runtime_info())

        if options.enable_clip:
            with recorder.stage("clip_embedding", model=self.settings.clip_model_name):
                try:
                    embedder = CLIPSceneEmbedder(
                        model_name=self.settings.clip_model_name,
                        device=self.settings.device,
                    )
                    for scene in scenes:
                        if scene.representative_frame:
                            scene.clip_embedding = embedder.embed_image(scene.representative_frame)
                except Exception as exc:
                    for scene in scenes:
                        scene.metadata["clip_error"] = str(exc)

        benchmark = recorder.metrics(total_frames=len(frames))
        metadata = VideoMetadata(
            video_id=job_id,
            source=source,
            local_video_path=str(local_video_path),
            duration=duration,
            config=PipelineConfig(
                fps=options.fps,
                scene_detector=options.scene_detector,
                classifier=options.classifier,
                temporal_window_size=options.temporal_window_size,
                temporal_stride=options.temporal_stride,
                clip_enabled=options.enable_clip,
                quality_scoring_enabled=options.enable_quality_scoring,
                multi_gpu_enabled=options.enable_multi_gpu,
            ),
            frames=frames,
            temporal_windows=temporal_windows,
            scenes=scenes,
            benchmark=benchmark,
        )

        metadata_path = job_dir / "metadata.json"
        metadata.metadata_path = str(metadata_path)
        metadata_path.write_text(
            json.dumps(metadata.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        return metadata
