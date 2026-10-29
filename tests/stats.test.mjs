import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { median, summarize } from '../src/lib/stats.mjs';

test('median odd and even', () => {
  assert.equal(median([9, 1, 4]), 4);
  assert.equal(median([9, 1, 4, 2]), 3);
  assert.equal(median([]), null);
});

test('summarize sorted quantiles, filtering invalid values', () => {
  assert.deepEqual(summarize([5, 1, 2, 3, 4, Number.NaN]), {
    n: 5, min: 1, median: 3, p90: 5, max: 5, mean: 3,
  });
  assert.equal(summarize([]), null);
});
