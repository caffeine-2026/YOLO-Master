const assert = require("assert");

const { MODEL_CATALOG } = require("../models/catalog");
const { letterboxRGBA } = require("../utils/preprocess");
const { decodeOutput, intersectionOverUnion } = require("../utils/postprocess");
const { summarize } = require("../utils/time");

function testCatalog() {
  assert.strictEqual(MODEL_CATALOG.length, 2);
  MODEL_CATALOG.forEach((model) => {
    assert.strictEqual(model.labels.length, 6);
    assert.strictEqual(model.inputSize, 640);
    assert.ok(model.file.endsWith(".onnx"));
  });
}

function testPreprocess() {
  const pixels = new Uint8Array([
    255, 0, 0, 255,
    0, 255, 0, 255,
    0, 0, 255, 255,
    255, 255, 255, 255,
  ]);
  const output = letterboxRGBA({ data: pixels.buffer, width: 2, height: 2 }, 4);
  assert.deepStrictEqual(output.data.length, 3 * 4 * 4);
  assert.strictEqual(output.transform.scale, 2);
  assert.strictEqual(output.data[0], 1);
  assert.strictEqual(output.data[16], 0);
  assert.strictEqual(output.data[32], 0);
}

function testPostprocess() {
  const values = new Float32Array([
    2.0, 2.1,
    2.0, 2.1,
    2.0, 2.0,
    2.0, 2.0,
    0.9, 0.8,
    0.1, 0.2,
  ]);
  const output = { data: values.buffer, shape: [1, 6, 2], type: "float32" };
  const detections = decodeOutput(
    output,
    ["defect", "other"],
    { inputSize: 4, sourceWidth: 4, sourceHeight: 4, scale: 1, left: 0, top: 0 },
    0.25,
    0.45,
  );
  assert.strictEqual(detections.length, 1);
  assert.strictEqual(detections[0].label, "defect");
  assert.ok(Math.abs(detections[0].confidence - 0.9) < 1e-6);
  assert.ok(intersectionOverUnion({ x1: 0, y1: 0, x2: 2, y2: 2 }, { x1: 1, y1: 1, x2: 3, y2: 3 }) > 0.14);
}

function testMetrics() {
  const summary = summarize([10, 20, 30, 40]);
  assert.strictEqual(summary.mean, 25);
  assert.strictEqual(summary.p50, 20);
  assert.strictEqual(summary.p95, 40);
  assert.strictEqual(summary.fps, 40);
}

testCatalog();
testPreprocess();
testPostprocess();
testMetrics();
console.log("C3 Mini Program core tests: PASS");
