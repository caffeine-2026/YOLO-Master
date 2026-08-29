import type * as Ort from 'onnxruntime-web/all';

import { prepareCanvasSource } from '@/lib/image';
import { ensureModelBytes } from '@/lib/model-cache';
import type { ModelSpec } from '@/lib/models';
import { decodeOutput, type Detection } from '@/lib/postprocess';

export type InferenceTimings = {
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  totalMs: number;
};

export type InferenceResult = {
  detections: Detection[];
  timings: InferenceTimings;
  width: number;
  height: number;
};

type Backend = 'webgpu' | 'webgl' | 'wasm';

function backendCandidates(): Backend[] {
  const backends: Backend[] = [];
  if ('gpu' in navigator) backends.push('webgpu');
  const canvas = document.createElement('canvas');
  if (canvas.getContext('webgl2') || canvas.getContext('webgl')) backends.push('webgl');
  backends.push('wasm');
  return backends;
}

export class EdgeInferenceEngine {
  private ort: typeof Ort | null = null;
  private session: Ort.InferenceSession | null = null;
  private model: ModelSpec | null = null;
  private inputCanvas: HTMLCanvasElement | null = null;
  backend: Backend | null = null;

  async load(model: ModelSpec, onProgress: (progress: number) => void = () => undefined) {
    if (this.session && this.model?.id === model.id) return { backend: this.backend, source: 'memory' as const };
    this.release();
    const artifact = await ensureModelBytes(model, onProgress);
    const ort = await import('onnxruntime-web/all');
    ort.env.wasm.numThreads = globalThis.crossOriginIsolated
      ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2))
      : 1;
    ort.env.wasm.proxy = false;
    const failures: string[] = [];
    for (const backend of backendCandidates()) {
      try {
        const session = await ort.InferenceSession.create(new Uint8Array(artifact.bytes), {
          executionProviders: [backend],
          graphOptimizationLevel: 'all',
        });
        this.ort = ort;
        this.session = session;
        this.model = model;
        this.backend = backend;
        this.inputCanvas = document.createElement('canvas');
        return { backend, source: artifact.source };
      } catch (error) {
        failures.push(`${backend}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    throw new Error(`No browser inference backend could load this model. ${failures.join(' · ')}`);
  }

  async runTensor(data: Float32Array): Promise<{ output: Ort.Tensor; inferenceMs: number }> {
    if (!this.session || !this.model || !this.ort) throw new Error('Load a model before inference.');
    const tensor = new this.ort.Tensor('float32', data, [1, 3, this.model.inputSize, this.model.inputSize]);
    const started = performance.now();
    const outputMap = await this.session.run({ [this.model.inputName]: tensor });
    const inferenceMs = performance.now() - started;
    const output = outputMap[this.model.outputName];
    if (!output) throw new Error(`Model output '${this.model.outputName}' was not returned.`);
    return { output, inferenceMs };
  }

  async runSource(
    source: CanvasImageSource,
    width: number,
    height: number,
    confidence = 0.25,
    iou = 0.45,
  ): Promise<InferenceResult> {
    if (!this.model || !this.inputCanvas) throw new Error('Load a model before inference.');
    const totalStarted = performance.now();
    const preprocessStarted = performance.now();
    const prepared = prepareCanvasSource(source, width, height, this.model.inputSize, this.inputCanvas);
    const preprocessMs = performance.now() - preprocessStarted;
    const inference = await this.runTensor(prepared.data);
    const postprocessStarted = performance.now();
    const detections = decodeOutput(
      inference.output.data as ArrayLike<number>,
      inference.output.dims,
      this.model.labels,
      prepared.transform,
      confidence,
      iou,
    );
    const postprocessMs = performance.now() - postprocessStarted;
    return {
      detections,
      width,
      height,
      timings: {
        preprocessMs,
        inferenceMs: inference.inferenceMs,
        postprocessMs,
        totalMs: performance.now() - totalStarted,
      },
    };
  }

  release(): void {
    this.session?.release();
    this.session = null;
    this.model = null;
    this.inputCanvas = null;
    this.backend = null;
  }
}
