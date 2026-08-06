import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

export default function AccuracyChart({ history }) {
  const data = [...(history || [])]
    .reverse()
    .map((r) => ({
      trained_at: new Date(r.trained_at).toLocaleString(),
      accuracy: Number(r.accuracy_frozen_test) * 100,
      promoted: r.promoted,
      n_live_samples: r.n_live_samples,
    }));

  return (
    <div style={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 10, padding: 16, marginBottom: 24 }}>
      <h3 style={{ marginTop: 0 }}>Classifier accuracy over time (frozen held-out set)</h3>
      <p style={{ fontSize: 12, color: "#8b949e", marginTop: -8 }}>
        Each point is one retrain run (classifier/retrain_from_logs.py), evaluated on the same
        frozen bootstrap split every time — comparable across runs regardless of what live
        traffic looked like that week.
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
          <XAxis dataKey="trained_at" stroke="#8b949e" tick={{ fontSize: 10 }} />
          <YAxis domain={[0, 100]} stroke="#8b949e" />
          <Tooltip contentStyle={{ background: "#0d1117", border: "1px solid #30363d" }} />
          <Legend />
          <Line type="monotone" dataKey="accuracy" stroke="#3fb950" name="Accuracy (%)" dot />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
