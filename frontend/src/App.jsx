import {
  Activity,
  BarChart3,
  Film,
  Gauge,
  Loader2,
  Play,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  UploadCloud,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

function App() {
  const [source, setSource] = useState("");
  const [fps, setFps] = useState(1);
  const [classifier, setClassifier] = useState("resnet50");
  const [detector, setDetector] = useState("histogram");
  const [clipEnabled, setClipEnabled] = useState(false);
  const [qualityEnabled, setQualityEnabled] = useState(true);
  const [multiGpuEnabled, setMultiGpuEnabled] = useState(false);
  const [job, setJob] = useState(null);
  const [status, setStatus] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    refreshMetrics();
    const id = window.setInterval(refreshMetrics, 5000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!job?.job_id) return undefined;
    const id = window.setInterval(() => refreshJob(job.job_id), 2500);
    refreshJob(job.job_id);
    return () => window.clearInterval(id);
  }, [job?.job_id]);

  const scenes = metadata?.scenes ?? [];
  const stageRows = metadata?.benchmark?.stage_latencies ?? [];
  const completed = status?.status === "completed";
  const qualityScores = scenes
    .map((scene) => scene.quality?.data_quality_score)
    .filter((score) => typeof score === "number");
  const averageQuality = qualityScores.length
    ? qualityScores.reduce((total, score) => total + score, 0) / qualityScores.length
    : metrics?.average_quality_score;
  const lowQualityScenes = scenes.filter(
    (scene) => (scene.quality?.data_quality_score ?? 1) < 0.55,
  ).length;

  const topLabels = useMemo(() => {
    const counts = new Map();
    scenes.forEach((scene) => {
      const label = scene.labels?.[0]?.label;
      if (label) counts.set(label, (counts.get(label) ?? 0) + 1);
    });
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
  }, [scenes]);

  async function submitVideo(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setMetadata(null);
    setSearchResults([]);
    try {
      const response = await fetch(`${API_BASE}/videos`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source,
          fps: Number(fps),
          classifier,
          scene_detector: detector,
          enable_clip: clipEnabled,
          enable_quality_scoring: qualityEnabled,
          enable_multi_gpu: multiGpuEnabled,
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      setJob(payload);
      setStatus(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function refreshJob(jobId) {
    try {
      const response = await fetch(`${API_BASE}/jobs/${jobId}`);
      if (!response.ok) return;
      const payload = await response.json();
      setStatus(payload);
      if (payload.status === "completed") {
        const metadataResponse = await fetch(`${API_BASE}/videos/${jobId}/metadata`);
        if (metadataResponse.ok) setMetadata(await metadataResponse.json());
      }
    } catch (err) {
      setError(err.message);
    }
  }

  async function refreshMetrics() {
    try {
      const response = await fetch(`${API_BASE}/metrics`);
      if (response.ok) setMetrics(await response.json());
    } catch {
      setMetrics(null);
    }
  }

  async function runSearch(event) {
    event.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError("");
    try {
      const params = new URLSearchParams({ query, top_k: "8" });
      if (job?.job_id) params.set("job_id", job.job_id);
      const response = await fetch(`${API_BASE}/search?${params.toString()}`);
      if (!response.ok) throw new Error(await response.text());
      setSearchResults(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Scene Understanding</p>
          <h1>Video Pipeline Dashboard</h1>
        </div>
        <StatusPill status={status?.status ?? "idle"} />
      </header>

      <section className="workspace">
        <aside className="control-panel">
          <form onSubmit={submitVideo} className="panel-section">
            <label>
              Source
              <input
                value={source}
                onChange={(event) => setSource(event.target.value)}
                placeholder="/path/video.mp4 or YouTube URL"
                required
              />
            </label>

            <div className="split">
              <label>
                FPS
                <input
                  type="number"
                  min="0.1"
                  step="0.1"
                  value={fps}
                  onChange={(event) => setFps(event.target.value)}
                />
              </label>
              <label>
                Classifier
                <select value={classifier} onChange={(event) => setClassifier(event.target.value)}>
                  <option value="resnet50">ResNet50</option>
                  <option value="efficientnet_b0">EfficientNet-B0</option>
                </select>
              </label>
            </div>

            <label>
              Scene Detector
              <select value={detector} onChange={(event) => setDetector(event.target.value)}>
                <option value="histogram">Histogram Difference</option>
                <option value="pyscenedetect">PySceneDetect</option>
              </select>
            </label>

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={clipEnabled}
                onChange={(event) => setClipEnabled(event.target.checked)}
              />
              <span>CLIP embeddings</span>
            </label>

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={qualityEnabled}
                onChange={(event) => setQualityEnabled(event.target.checked)}
              />
              <span>Quality scoring</span>
            </label>

            <label className="toggle-row">
              <input
                type="checkbox"
                checked={multiGpuEnabled}
                onChange={(event) => setMultiGpuEnabled(event.target.checked)}
              />
              <span>DataParallel GPU</span>
            </label>

            <button type="submit" disabled={busy || !source.trim()}>
              {busy ? <Loader2 className="spin" size={18} /> : <UploadCloud size={18} />}
              Submit
            </button>
          </form>

          <form onSubmit={runSearch} className="panel-section">
            <label>
              Semantic Search
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="night street, cooking, audience..."
              />
            </label>
            <button type="submit" disabled={busy || !query.trim()}>
              <Search size={18} />
              Search
            </button>
          </form>

          {error && <div className="error-box">{error}</div>}
        </aside>

        <section className="content-area">
          <div className="metric-grid">
            <Metric icon={<Film />} label="Scenes" value={scenes.length || metrics?.total_scenes || 0} />
            <Metric
              icon={<Gauge />}
              label="Frames/sec"
              value={formatNumber(metadata?.benchmark?.frames_per_second ?? metrics?.average_frames_per_second)}
            />
            <Metric
              icon={<Activity />}
              label="Latency ms"
              value={formatNumber(metadata?.benchmark?.total_latency_ms ?? metrics?.average_total_latency_ms)}
            />
            <Metric icon={<BarChart3 />} label="Jobs" value={metrics?.total_jobs ?? 0} />
            <Metric
              icon={<ShieldCheck />}
              label="Quality"
              value={`${Math.round((averageQuality ?? 0) * 100)}%`}
            />
            <Metric
              icon={<SlidersHorizontal />}
              label="Review Scenes"
              value={scenes.length ? lowQualityScenes : metrics?.low_quality_scenes || 0}
            />
          </div>

          {completed && (
            <div className="summary-band">
              <div>
                <h2>{metadata?.source}</h2>
                <p>
                  {metadata?.frames?.length ?? 0} frames, {scenes.length} scenes, detector{" "}
                  {metadata?.config?.scene_detector}
                </p>
              </div>
              <div className="label-stack">
                {metadata?.config?.multi_gpu_enabled && <span>DataParallel ready</span>}
                {topLabels.map(([label, count]) => (
                  <span key={label}>
                    {label} <strong>{count}</strong>
                  </span>
                ))}
              </div>
            </div>
          )}

          <section className="stage-table">
            <h2>Stage Latency</h2>
            {stageRows.length ? (
              <table>
                <thead>
                  <tr>
                    <th>Stage</th>
                    <th>Latency</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {stageRows.map((stage) => (
                    <tr key={stage.name}>
                      <td>{stage.name}</td>
                      <td>{formatNumber(stage.latency_ms)} ms</td>
                      <td>{Object.entries(stage.details ?? {}).map(([key, value]) => `${key}: ${value}`).join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <EmptyState icon={<Play />} text="Submit a video to populate latency metrics." />
            )}
          </section>

          {scenes.length > 0 && (
            <section className="quality-board">
              <h2>Scene Quality</h2>
              <div className="quality-list">
                {scenes.map((scene) => (
                  <QualityRow key={scene.scene_id} scene={scene} />
                ))}
              </div>
            </section>
          )}

          {searchResults.length > 0 && (
            <section className="scene-grid-wrap">
              <h2>Search Results</h2>
              <div className="scene-grid">
                {searchResults.map((scene) => (
                  <SceneTile key={scene.scene_id} scene={scene} score={scene.score} />
                ))}
              </div>
            </section>
          )}

          <section className="scene-grid-wrap">
            <h2>Detected Scenes</h2>
            {scenes.length ? (
              <div className="scene-grid">
                {scenes.map((scene) => (
                  <SceneTile key={scene.scene_id} scene={scene} />
                ))}
              </div>
            ) : (
              <EmptyState icon={<Film />} text="Scene thumbnails will appear after processing." />
            )}
          </section>
        </section>
      </section>
    </main>
  );
}

function Metric({ icon, label, value }) {
  return (
    <div className="metric">
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value ?? 0}</strong>
    </div>
  );
}

function SceneTile({ scene, score }) {
  const label = scene.labels?.[0];
  const quality = scene.quality?.data_quality_score;
  return (
    <article className="scene-tile">
      <div className="thumb">
        {scene.representative_frame ? (
          <img src={assetUrl(scene.representative_frame)} alt="" />
        ) : (
          <Film size={28} />
        )}
      </div>
      <div className="scene-body">
        <div className="scene-title">
          <strong>Scene {scene.scene_index + 1}</strong>
          <span>
            {formatNumber(scene.start_timestamp)}s - {formatNumber(scene.end_timestamp)}s
          </span>
        </div>
        <p>{label ? `${label.label} (${Math.round(label.confidence * 100)}%)` : "No label"}</p>
        {typeof quality === "number" && (
          <div className="quality-strip">
            <span>{scene.quality.quality_grade}</span>
            <meter min="0" max="1" value={quality} />
            <strong>{Math.round(quality * 100)}%</strong>
          </div>
        )}
        {typeof score === "number" && <meter min="0" max="1" value={Math.max(score, 0)} />}
      </div>
    </article>
  );
}

function QualityRow({ scene }) {
  const quality = scene.quality;
  if (!quality) return null;
  return (
    <div className="quality-row">
      <div>
        <strong>Scene {scene.scene_index + 1}</strong>
        <span>{quality.recommended_action}</span>
      </div>
      <meter min="0" max="1" value={quality.data_quality_score} />
      <strong>{Math.round(quality.data_quality_score * 100)}%</strong>
      <p>{quality.flags.length ? quality.flags.join(", ") : "clean"}</p>
    </div>
  );
}

function StatusPill({ status }) {
  return <span className={`status status-${status}`}>{status}</span>;
}

function EmptyState({ icon, text }) {
  return (
    <div className="empty-state">
      {icon}
      <span>{text}</span>
    </div>
  );
}

function assetUrl(path) {
  if (!path) return "";
  if (path.startsWith("http") || path.startsWith("/artifacts")) return `${API_BASE}${path}`;
  if (path.startsWith("artifacts/")) return `${API_BASE}/${path}`;
  const marker = "/artifacts/";
  const index = path.indexOf(marker);
  if (index >= 0) return `${API_BASE}${path.slice(index)}`;
  return path;
}

function formatNumber(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "0";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export default App;
