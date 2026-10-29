export const MODEL_SIZE = 320;

/** The same transform is used by letterboxing and by box reverse-projection. */
export function letterboxGeometry(width, height, size = MODEL_SIZE) {
  if (!(width > 0 && height > 0 && size > 0)) {
    throw new Error('Image dimensions must be positive');
  }
  const nominalScale = Math.min(size / width, size / height);
  const drawWidth = Math.min(size, Math.max(1, Math.round(width * nominalScale)));
  const drawHeight = Math.min(size, Math.max(1, Math.round(height * nominalScale)));
  const padX = Math.floor((size - drawWidth) / 2);
  const padY = Math.floor((size - drawHeight) / 2);
  return {
    size,
    sourceWidth: width, sourceHeight: height,
    drawWidth, drawHeight, padX, padY,
    scaleX: drawWidth / width, scaleY: drawHeight / height,
  };
}

export function projectBox(box, g) {
  const x1 = Math.max(0, Math.min(g.sourceWidth, (box.x1 - g.padX) / g.scaleX));
  const y1 = Math.max(0, Math.min(g.sourceHeight, (box.y1 - g.padY) / g.scaleY));
  const x2 = Math.max(0, Math.min(g.sourceWidth, (box.x2 - g.padX) / g.scaleX));
  const y2 = Math.max(0, Math.min(g.sourceHeight, (box.y2 - g.padY) / g.scaleY));
  return { ...box, x1, y1, x2, y2 };
}
