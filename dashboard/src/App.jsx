import { useEffect, useState } from "react";
import { api } from "./api";
import MetricsSummary from "./components/MetricsSummary.jsx";
import BackendVolumeChart from "./components/BackendVolumeChart.jsx";
import AccuracyChart from "./components/AccuracyChart.jsx";
import QueryLogTable from "./components/QueryLogTable.jsx";

const POLL_MS = 10000;

export default function App() {
  const [metrics, setMetrics] = useState(null);
  const [accuracyHistory, setAccuracyHistory] = useState([]);
  const [logRows, setLogRows] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const [m, acc, log] = await Promise.all([
          api.metrics(24),
          api.classifierAccuracy(50),
          api.queryLog(50),
        ]);
        if (!cancelled) {
          setMetrics(m);
          setAccuracyHistory(acc);
          setLogRows(log);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    }

    poll();
    const interval = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", background: "#0d1117", color: "#e6edf3", minHeight: "100vh", padding: 24 }}>
      <h1>Multi-Database Query Router — Dashboard</h1>
      <p style={{ color: "#8b949e", marginTop: -8 }}>
        Live view of routing decisions, router-vs-naive-baseline performance, and classifier
        accuracy over retrains. Refreshes every {POLL_MS / 1000}s.
      </p>
      {error && (
        <div style={{ background: "#3d1518", border: "1px solid #f85149", borderRadius: 8, padding: 12, marginBottom: 16 }}>
          Couldn't reach the router API: {error}. Is it running at{" "}
          {import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"}?
        </div>
      )}
      <MetricsSummary metrics={metrics} />
      <BackendVolumeChart byBackend={metrics?.router_by_backend} />
      <AccuracyChart history={accuracyHistory} />
      <QueryLogTable rows={logRows} />
    </div>
  );
}
