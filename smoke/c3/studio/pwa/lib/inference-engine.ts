import type * as Ort from 'onnxruntime-web';

import { backendCandidatesForRuntime, type EdgeBackend } from '@/lib/backend-order';
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

function backendCandidates(): EdgeBackend[] {
  const canvas = document.createElement('canvas');
  const hasWebGl = Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'));
  return backendCandidatesForRuntime(navigator, hasWebGl);
}

export class EdgeInferenceEngine {
  private ort: typeof Ort | null = null;
  private session: Ort.InferenceSession | null = null;
  private model: ModelSpec | null = null;
  private modelBytes: ArrayBuffer | null = null;
  private inputCanvas: HTMLCanvasElement | null = null;
  backend: EdgeBackend | null = null;

  private async createSession(backend: EdgeBackend, bytes: ArrayBuffer) {
    const ort: typeof Ort = backend === 'webgl'
      ? await import('onnxruntime-web/webgl')
      : await import('onnxruntime-web/wasm');
    if (backend === 'wasm') {
      ort.env.wasm.numThreads = globalThis.crossOriginIsolated
        ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2))
        : 1;
      ort.env.wasm.proxy = false;
    }
    const session = await ort.InferenceSession.create(new Uint8Array(bytes), {
      executionProviders: [backend],
      graphOptimizationLevel: 'all',
    });
    return { ort, session };
  }

  async load(model: ModelSpec, onProgress: (progress: number) => void = () => undefined) {
    if (this.session && this.model?.id === model.id) return { backend: this.backend, source: 'memory' as const };
    this.release();
    const artifact = await ensureModelBytes(model, onProgress);
    const failures: string[] = [];
    for (const backend of backendCandidates()) {
      try {
        const { ort, session } = await this.createSession(backend, artifact.bytes);
        this.ort = ort;
        this.session = session;
        this.model = model;
        this.modelBytes = artifact.bytes;
        this.backend = backend;
        this.inputCanvas = document.createElement('canvas');
        return { backend, source: artifact.source };
      } catch (error) {
        failures.push(`${backend}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }
    throw new Error(`No browser inference backend could load this model. ${failures.join(' | ')}`);
  }

  async runTensor(data: Float32Array): Promise<{ output: Ort.Tensor; inferenceMs: number }> {
    if (!this.session || !this.model || !this.ort) throw new Error('Load a model before inference.');
    const execute = async () => {
      const tensor = new this.ort!.Tensor('float32', data, [1, 3, this.model!.inputSize, this.model!.inputSize]);
      const started = performance.now();
      const outputMap = await this.session!.run({ [this.model!.inputName]: tensor });
      const inferenceMs = performance.now() - started;
      const output = outputMap[this.model!.outputName];
      if (!output) throw new Error(`Model output '${this.model!.outputName}' was not returned.`);
      return { output, inferenceMs };
    };

    try {
      return await execute();
    } catch (error) {
      if (this.backend !== 'webgl' || !this.modelBytes) throw error;
      const previousSession = this.session;
      const { ort, session } = await this.createSession('wasm', this.modelBytes);
      this.ort = ort;
      this.session = session;
      this.backend = 'wasm';
      previousSession.release();
      return execute();
    }
  }

  async runSource(
    source: CanvasImageSource,
    width: number,
    height: number,
    confidence: number,
    iou = 0.45,
  ): Promise<InferenceResult> {
    if (!this.model || !this.inputCanvas) throw new Error('Load a model before inference.');
    const totalStarted = performance.now();
    const preprocessStarted = performance.now();
    const prepared = prepareCanvasSource(
      source,
      width,
      height,
      this.model.inputSize,
      this.inputCanvas,
      this.model.inputColorMode === 'grayscale',
    );
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
    this.modelBytes = null;
    this.inputCanvas = null;
    this.backend = null;
  }
}
