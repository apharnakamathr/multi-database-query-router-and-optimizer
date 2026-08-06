export default function MetricsSummary({ metrics }) {
  if (!metrics) return null;
  const { router_overall, naive_baseline_overall, router_vs_baseline, window_hours } = metrics;

  const card = (label, value, sub) => (
    <div style={styles.card}>
      <div style={styles.cardLabel}>{label}</div>
      <div style={styles.cardValue}>{value}</div>
      {sub && <div style={styles.cardSub}>{sub}</div>}
    </div>
  );

  const fmtMs = (v) => (v == null ? "—" : `${Number(v).toFixed(1)} ms`);
  const fmtUsd = (v) => (v == null ? "—" : `$${Number(v).toFixed(4)}`);
  const fmtPct = (v) => (v == null ? "—" : `${Number(v).toFixed(1)}%`);

  return (
    <div>
      <h3 style={{ marginBottom: 8 }}>Last {window_hours}h — Router vs naive baseline</h3>
      <div style={styles.row}>
        {card("Router avg latency", fmtMs(router_overall?.avg_latency_ms), `${router_overall?.n || 0} queries`)}
        {card("Baseline avg latency", fmtMs(naive_baseline_overall?.avg_latency_ms), `${naive_baseline_overall?.n || 0} queries`)}
        {card("Latency improvement", fmtPct(router_vs_baseline?.latency_improvement_pct), "router vs. naive")}
        {card("Router total cost", fmtUsd(router_overall?.total_cost_usd))}
        {card("Baseline total cost", fmtUsd(naive_baseline_overall?.total_cost_usd))}
        {card("Cost improvement", fmtPct(router_vs_baseline?.cost_improvement_pct), "router vs. naive")}
      </div>
    </div>
  );
}

const styles = {
  row: { display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 24 },
  card: {
    background: "#161b22",
    border: "1px solid #30363d",
    borderRadius: 10,
    padding: "12px 16px",
    minWidth: 160,
  },
  cardLabel: { fontSize: 12, color: "#8b949e", marginBottom: 4 },
  cardValue: { fontSize: 22, fontWeight: 600, color: "#e6edf3" },
  cardSub: { fontSize: 11, color: "#6e7681", marginTop: 2 },
};
