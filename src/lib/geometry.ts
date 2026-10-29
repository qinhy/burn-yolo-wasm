import { letterboxGeometry, projectBox } from './geometry.mjs';
import type { Detection } from './inference';

export interface Geometry {
  size: number;
  sourceWidth: number; sourceHeight: number;
  drawWidth: number; drawHeight: number;
  padX: number; padY: number;
  scaleX: number; scaleY: number;
}

export { letterboxGeometry, projectBox };

export function preprocess(
  image: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
): { rgba: Uint8Array; geometry: Geometry } {
  const geometry: Geometry = letterboxGeometry(sourceWidth, sourceHeight);
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = geometry.size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2D Canvas unavailable');
  ctx.fillStyle = '#727272';
  ctx.fillRect(0, 0, geometry.size, geometry.size);
  ctx.drawImage(image, geometry.padX, geometry.padY, geometry.drawWidth, geometry.drawHeight);
  const imageData = ctx.getImageData(0, 0, geometry.size, geometry.size);
  return { rgba: new Uint8Array(imageData.data), geometry };
}

export function drawDetections(
  canvas: HTMLCanvasElement,
  source: CanvasImageSource,
  width: number,
  height: number,
  detections: Detection[],
  geometry: Geometry,
  classNames: string[],
): void {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.drawImage(source, 0, 0, width, height);
  const lineWidth = Math.max(2, Math.round(width / 420));
  ctx.lineWidth = lineWidth;
  ctx.font = `${Math.max(14, Math.round(width / 45))}px system-ui, sans-serif`;

  for (const d of detections) {
    const b = projectBox(d, geometry);
    if (b.x2 <= b.x1 || b.y2 <= b.y1) continue;
    const label = `${classNames[d.class_id] ?? `class ${d.class_id}`} ${(d.score * 100).toFixed(0)}%`;
    const hue = (d.class_id * 47 + 187) % 360;
    const color = `hsl(${hue} 88% 57%)`;
    ctx.strokeStyle = color;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    const metrics = ctx.measureText(label);
    const fontHeight = Math.max(14, Math.round(width / 45));
    const labelWidth = metrics.width + 14;
    const labelY = b.y1 >= fontHeight + 10 ? b.y1 - fontHeight - 8 : b.y1;
    ctx.fillStyle = color;
    ctx.fillRect(b.x1, labelY, labelWidth, fontHeight + 8);
    ctx.fillStyle = '#0b1421';
    ctx.fillText(label, b.x1 + 7, labelY + fontHeight + 1);
  }
}
