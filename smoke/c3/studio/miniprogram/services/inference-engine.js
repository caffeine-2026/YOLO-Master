const { ensureModel } = require("./model-store");
const { letterboxRGBA } = require("../utils/preprocess");
const { decodeOutput } = require("../utils/postprocess");
const { nowMs } = require("../utils/time");

function inferenceEnvironment() {
  return new Promise((resolve) => {
    if (!wx.getInferenceEnvInfo) {
      resolve({ supported: false, version: "unavailable", reason: "WeChat base library is below 2.30.0" });
      return;
    }
    wx.getInferenceEnvInfo({
      success: (result) => resolve({ supported: true, version: String(result.ver || "available"), raw: result }),
      fail: (error) => resolve({ supported: false, version: "unavailable", reason: error.errMsg || "Not supported" }),
    });
  });
}

class EdgeInferenceEngine {
  constructor() {
    this.session = null;
    this.model = null;
    this.environment = null;
  }

  async load(model, onProgress = () => {}) {
    this.destroy();
    this.environment = await inferenceEnvironment();
    if (!this.environment.supported || !wx.createInferenceSession) {
      throw new Error(this.environment.reason || "Native ONNX inference is unavailable on this device");
    }
    const local = await ensureModel(model, onProgress);
    const session = wx.createInferenceSession({
      model: local.path,
      precisionLevel: 0,
      allowNPU: true,
      allowQuantize: false,
      typicalShape: { [model.inputName]: [1, 3, model.inputSize, model.inputSize] },
    });

    await new Promise((resolve, reject) => {
      let settled = false;
      const timer = setTimeout(() => {
        if (!settled) reject(new Error("Timed out while WeChat was compiling the ONNX model"));
      }, 90000);
      session.onLoad(() => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve();
      });
      session.onError((error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        reject(new Error(error.errMsg || "WeChat inference session failed to load"));
      });
    });

    this.session = session;
    this.model = model;
    return { environment: this.environment, modelPath: local.path, digest: local.digest };
  }

  async runTensor(tensor) {
    if (!this.session || !this.model) throw new Error("Load a model before inference");
    const started = nowMs();
    const result = await this.session.run({
      [this.model.inputName]: {
        data: tensor.buffer,
        shape: [1, 3, this.model.inputSize, this.model.inputSize],
        type: "float32",
      },
    });
    const output = result[this.model.outputName];
    if (!output) throw new Error(`Model output '${this.model.outputName}' was not returned`);
    return { output, latencyMs: nowMs() - started };
  }

  async runRGBA(frame, options = {}) {
    if (!this.model) throw new Error("Load a model before inference");
    const preprocessStarted = nowMs();
    const prepared = letterboxRGBA(frame, this.model.inputSize);
    const preprocessMs = nowMs() - preprocessStarted;
    const inference = await this.runTensor(prepared.data);
    const postprocessStarted = nowMs();
    const detections = decodeOutput(
      inference.output,
      this.model.labels,
      prepared.transform,
      Number(options.confidence || 0.25),
      Number(options.iou || 0.45),
    );
    const postprocessMs = nowMs() - postprocessStarted;
    return {
      detections,
      transform: prepared.transform,
      timings: {
        preprocessMs,
        inferenceMs: inference.latencyMs,
        postprocessMs,
        totalMs: preprocessMs + inference.latencyMs + postprocessMs,
      },
    };
  }

  destroy() {
    if (this.session) {
      try {
        this.session.destroy();
      } catch (error) {
        console.warn("Unable to destroy inference session", error);
      }
    }
    this.session = null;
    this.model = null;
  }
}

module.exports = { EdgeInferenceEngine, inferenceEnvironment };
