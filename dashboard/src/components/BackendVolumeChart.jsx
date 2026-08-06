import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

const COLORS = { n: "#58a6ff", avg_latency_ms: "#f78166" };

export default function BackendVolumeChart({ byBackend }) {
  const data = (byBackend || []).map((r) => ({
    backend: r.backend,
    queries: Number(r.n),
    avg_latency_ms: Number(r.avg_latency_ms).toFixed(1),
    fallback_rate: (Number(r.fallback_rate) * 100).toFixed(1),
  }));

  return (
    <div style={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 10, padding: 16, marginBottom: 24 }}>
      <h3 style={{ marginTop: 0 }}>Query volume & latency by backend</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
          <XAxis dataKey="backend" stroke="#8b949e" />
          <YAxis yAxisId="left" stroke="#8b949e" />
          <YAxis yAxisId="right" orientation="right" stroke="#8b949e" />
          <Tooltip contentStyle={{ background: "#0d1117", border: "1px solid #30363d" }} />
          <Legend />
          <Bar yAxisId="left" dataKey="queries" fill={COLORS.n} name="Queries" />
          <Bar yAxisId="right" dataKey="avg_latency_ms" fill={COLORS.avg_latency_ms} name="Avg latency (ms)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
