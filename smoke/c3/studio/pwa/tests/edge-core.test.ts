import assert from 'node:assert/strict';
import test from 'node:test';

import { backendCandidatesForRuntime } from '../lib/backend-order.ts';
import { normalizedLuminance } from '../lib/image.ts';
import { summarizeLatencies } from '../lib/metrics.ts';
import { MODEL_CATALOG } from '../lib/models.ts';
import { decodeOutput, intersectionOverUnion, type Detection } from '../lib/postprocess.ts';

test('catalog pins two verified static models', () => {
  assert.equal(MODEL_CATALOG.length, 2);
  for (const model of MODEL_CATALOG) {
    assert.equal(model.labels.length, 6);
    assert.equal(model.inputSize, 640);
    assert.match(model.sha256, /^[a-f0-9]{64}$/);
    assert.match(model.url, /^\/models\/.+\.onnx$/);
    assert.ok(model.recommendedConfidence >= 0.3 && model.recommendedConfidence < 0.5);
    assert.ok(model.inputColorMode === 'rgb' || model.inputColorMode === 'grayscale');
    assert.ok(model.verifiedUse.length > 20);
    assert.ok(model.captureHint.length > 40);
    assert.ok(model.outOfScope.length > 30);
  }
});

test('NEU grayscale preprocessing uses one normalized luminance value for all channels', () => {
  assert.equal(normalizedLuminance(0, 0, 0), 0);
  assert.equal(normalizedLuminance(255, 255, 255), 1);
  assert.ok(Math.abs(normalizedLuminance(255, 0, 0) - 0.299) < 1e-12);
  assert.equal(MODEL_CATALOG.find((model) => model.dataset === 'NEU-DET')?.inputColorMode, 'grayscale');
});

test('Apple mobile browsers prefer the compatible WASM backend', () => {
  assert.deepEqual(
    backendCandidatesForRuntime({ userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)' }, true),
    ['wasm', 'webgl'],
  );
  assert.deepEqual(
    backendCandidatesForRuntime({ platform: 'MacIntel', maxTouchPoints: 5 }, true),
    ['wasm', 'webgl'],
  );
  assert.deepEqual(
    backendCandidatesForRuntime({ userAgent: 'Mozilla/5.0 (Linux; Android 15)' }, true),
    ['webgl', 'wasm'],
  );
});

test('channel-first YOLO output decodes with class-aware NMS', () => {
  const values = new Float32Array([
    2.0, 2.1,
    2.0, 2.1,
    2.0, 2.0,
    2.0, 2.0,
    0.9, 0.8,
    0.1, 0.2,
  ]);
  const detections = decodeOutput(
    values,
    [1, 6, 2],
    ['defect', 'other'],
    { inputSize: 4, sourceWidth: 4, sourceHeight: 4, scale: 1, left: 0, top: 0 },
    0.25,
    0.45,
  );
  assert.equal(detections.length, 1);
  assert.equal(detections[0].label, 'defect');
  assert.ok(Math.abs(detections[0].confidence - 0.9) < 1e-6);
});

test('IoU and latency summaries are deterministic', () => {
  const box = (x1: number, y1: number, x2: number, y2: number): Detection => ({
    classId: 0,
    label: 'defect',
    confidence: 1,
    x1,
    y1,
    x2,
    y2,
  });
  assert.ok(intersectionOverUnion(box(0, 0, 2, 2), box(1, 1, 3, 3)) > 0.14);
  const summary = summarizeLatencies([10, 20, 30, 40]);
  assert.equal(summary.mean, 25);
  assert.equal(summary.p50, 20);
  assert.equal(summary.p95, 40);
  assert.equal(summary.fps, 40);
});
