FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY scene_pipeline ./scene_pipeline
RUN pip install --no-cache-dir -e .

COPY scripts ./scripts
COPY schemas ./schemas
COPY reports ./reports

CMD ["uvicorn", "scene_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

