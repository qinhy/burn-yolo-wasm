export interface Summary {
  n: number;
  min: number;
  median: number;
  p90: number;
  max: number;
  mean: number;
}
export function median(values: number[]): number | null;
export function summarize(values: number[]): Summary | null;
