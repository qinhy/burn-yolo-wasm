import test from 'node:test';
import assert from 'node:assert/strict';
import { letterboxGeometry, projectBox } from '../src/lib/geometry.mjs';

test('square input fits without padding', () => {
  const g = letterboxGeometry(320, 320);
  assert.deepEqual([g.padX, g.padY, g.drawWidth, g.drawHeight], [0, 0, 320, 320]);
});

test('16:9 landscape image is padded vertically', () => {
  const g = letterboxGeometry(1920, 1080);
  assert.deepEqual([g.drawWidth, g.drawHeight, g.padX, g.padY], [320, 180, 0, 70]);
  const mapped = projectBox({ x1: 0, y1: 70, x2: 320, y2: 250 }, g);
  assert.deepEqual([mapped.x1, mapped.y1, mapped.x2, mapped.y2], [0, 0, 1920, 1080]);
});

test('portrait image is padded horizontally', () => {
  const g = letterboxGeometry(1080, 1920);
  assert.deepEqual([g.drawWidth, g.drawHeight, g.padX, g.padY], [180, 320, 70, 0]);
});

test('projected boxes stay within original image', () => {
  const g = letterboxGeometry(1024, 768);
  const b = projectBox({ x1: -99, y1: -50, x2: 400, y2: 500 }, g);
  assert.deepEqual([b.x1, b.y1, b.x2, b.y2], [0, 0, 1024, 768]);
});
