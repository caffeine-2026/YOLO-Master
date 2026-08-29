export type LatencySummary = {
  count: number;
  mean: number;
  p50: number;
  p95: number;
  min: number;
  max: number;
  fps: number;
};

export function summarizeLatencies(values: number[]): LatencySummary {
  if (!values.length) return { count: 0, mean: 0, p50: 0, p95: 0, min: 0, max: 0, fps: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const percentile = (ratio: number) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * ratio) - 1))];
  const mean = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;
  return {
    count: sorted.length,
    mean,
    p50: percentile(0.5),
    p95: percentile(0.95),
    min: sorted[0],
    max: sorted[sorted.length - 1],
    fps: mean > 0 ? 1000 / mean : 0,
  };
}
