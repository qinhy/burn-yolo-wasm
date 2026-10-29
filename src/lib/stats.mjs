/** Stable, dependency-free benchmark summaries: medians reduce outlier effects. */
export function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function summarize(values) {
  const valid = values.filter((value) => Number.isFinite(value));
  if (!valid.length) return null;
  const sorted = [...valid].sort((a, b) => a - b);
  return {
    n: sorted.length,
    min: sorted[0],
    median: median(sorted),
    p90: sorted[Math.ceil(sorted.length * 0.9) - 1],
    max: sorted[sorted.length - 1],
    mean: sorted.reduce((a, b) => a + b, 0) / sorted.length,
  };
}
