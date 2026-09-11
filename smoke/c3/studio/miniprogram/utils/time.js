function nowMs() {
  try {
    const performance = wx.getPerformance && wx.getPerformance();
    if (performance && typeof performance.now === "function") return performance.now();
  } catch (error) {
    // Date.now is sufficient as a compatibility fallback.
  }
  return Date.now();
}

function percentile(values, ratio) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * ratio) - 1));
  return sorted[index];
}

function summarize(values) {
  if (!values.length) return { count: 0, mean: 0, min: 0, max: 0, p50: 0, p95: 0, fps: 0 };
  const sum = values.reduce((total, value) => total + value, 0);
  const mean = sum / values.length;
  return {
    count: values.length,
    mean,
    min: Math.min(...values),
    max: Math.max(...values),
    p50: percentile(values, 0.5),
    p95: percentile(values, 0.95),
    fps: mean > 0 ? 1000 / mean : 0,
  };
}

module.exports = { nowMs, percentile, summarize };
