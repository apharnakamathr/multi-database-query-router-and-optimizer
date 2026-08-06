const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}`);
  return res.json();
}

export const api = {
  metrics: (hours = 24) => get(`/metrics?hours=${hours}`),
  queryLog: (limit = 50) => get(`/query-log?limit=${limit}`),
  classifierAccuracy: (limit = 50) => get(`/classifier-accuracy?limit=${limit}`),
  health: () => get("/health"),
};
