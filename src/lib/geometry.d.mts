export interface LetterboxGeometry {
  size: number;
  sourceWidth: number;
  sourceHeight: number;
  drawWidth: number;
  drawHeight: number;
  padX: number;
  padY: number;
  scaleX: number;
  scaleY: number;
}

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export function letterboxGeometry(width: number, height: number, size?: number): LetterboxGeometry;
export function projectBox<T extends Box>(box: T, g: LetterboxGeometry): T;
