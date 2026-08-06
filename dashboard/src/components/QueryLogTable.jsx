export default function QueryLogTable({ rows }) {
  return (
    <div style={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 10, padding: 16 }}>
      <h3 style={{ marginTop: 0 }}>Recent routing decisions</h3>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ textAlign: "left", color: "#8b949e" }}>
              <th style={th}>Query</th>
              <th style={th}>Predicted</th>
              <th style={th}>Actual</th>
              <th style={th}>Fallback</th>
              <th style={th}>Confidence</th>
              <th style={th}>Latency</th>
              <th style={th}>Source</th>
              <th style={th}>When</th>
            </tr>
          </thead>
          <tbody>
            {(rows || []).map((r) => (
              <tr key={r.log_id} style={{ borderTop: "1px solid #21262d" }}>
                <td style={td} title={r.query_text}>
                  {r.query_text.length > 50 ? r.query_text.slice(0, 50) + "…" : r.query_text}
                </td>
                <td style={td}>{r.predicted_backend}</td>
                <td style={td}>{r.actual_backend}</td>
                <td style={td}>{r.fallback ? "⚠️ yes" : "—"}</td>
                <td style={td}>{r.confidence != null ? Number(r.confidence).toFixed(2) : "—"}</td>
                <td style={td}>{Number(r.latency_ms).toFixed(1)} ms</td>
                <td style={td}>{r.source}</td>
                <td style={td}>{new Date(r.created_at).toLocaleTimeString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const th = { padding: "6px 10px", borderBottom: "1px solid #30363d" };
const td = { padding: "6px 10px", color: "#e6edf3" };
