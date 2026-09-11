const { EdgeInferenceEngine } = require("../../services/inference-engine");
const { shareReport } = require("../../utils/report");
const { summarize } = require("../../utils/time");

function fixed(number) {
  return Number(number || 0).toFixed(1);
}

Page({
  data: {
    models: [],
    modelTitles: [],
    selectedIndex: 0,
    iterationOptions: [30, 100],
    iterationIndex: 0,
    running: false,
    statusText: "",
    progressPercent: 0,
    errorText: "",
    hasResult: false,
    summary: {},
    deviceText: "",
    engineText: "",
  },

  onLoad() {
    const models = getApp().globalData.models;
    this.engine = new EdgeInferenceEngine();
    this.cancelRequested = false;
    this.report = null;
    this.setData({ models, modelTitles: models.map((model) => model.title) });
  },

  onUnload() {
    this.cancelRequested = true;
    if (this.engine) this.engine.destroy();
    wx.setKeepScreenOn({ keepScreenOn: false });
  },

  currentModel() {
    return this.data.models[this.data.selectedIndex];
  },

  onModelChange(event) {
    this.engine.destroy();
    this.setData({ selectedIndex: Number(event.detail.value), hasResult: false, errorText: "" });
  },

  onIterationChange(event) {
    this.setData({ iterationIndex: Number(event.detail.value) });
  },

  cancelBenchmark() {
    this.cancelRequested = true;
  },

  async runBenchmark() {
    const model = this.currentModel();
    const iterations = this.data.iterationOptions[this.data.iterationIndex];
    this.cancelRequested = false;
    this.setData({ running: true, hasResult: false, errorText: "", statusText: "Preparing model", progressPercent: 0 });
    wx.setKeepScreenOn({ keepScreenOn: true });
    try {
      const loaded = await this.engine.load(model, (progress) => this.setData({ statusText: `Downloading model ${progress}%` }));
      const input = new Float32Array(3 * model.inputSize * model.inputSize);
      input.fill(114 / 255);
      this.setData({ statusText: "Warmup 0 / 5" });
      for (let index = 0; index < 5; index += 1) {
        if (this.cancelRequested) throw new Error("Benchmark cancelled");
        await this.engine.runTensor(input);
        this.setData({ statusText: `Warmup ${index + 1} / 5` });
      }

      const latencies = [];
      for (let index = 0; index < iterations; index += 1) {
        if (this.cancelRequested) throw new Error("Benchmark cancelled");
        const output = await this.engine.runTensor(input);
        latencies.push(output.latencyMs);
        if ((index + 1) % 5 === 0 || index + 1 === iterations) {
          this.setData({
            statusText: `Measured ${index + 1} / ${iterations}`,
            progressPercent: Math.round(((index + 1) / iterations) * 100),
          });
        }
      }

      const rawSummary = summarize(latencies);
      const summary = {
        count: rawSummary.count,
        fps: fixed(rawSummary.fps),
        mean: fixed(rawSummary.mean),
        p50: fixed(rawSummary.p50),
        p95: fixed(rawSummary.p95),
        min: fixed(rawSummary.min),
        max: fixed(rawSummary.max),
      };
      const device = wx.getDeviceInfo ? wx.getDeviceInfo() : wx.getSystemInfoSync();
      const deviceText = `${device.brand || "unknown"} ${device.model || "device"} · ${device.system || device.platform || ""}`;
      this.report = {
        schema: "c3-edge-benchmark/v1",
        createdAt: new Date().toISOString(),
        model,
        engine: loaded.environment,
        configuration: { warmup: 5, iterations, inputShape: [1, 3, model.inputSize, model.inputSize], precisionLevel: 0, allowNPU: true },
        device,
        latenciesMs: latencies,
        summary: rawSummary,
        limitations: ["WeChat does not expose device thermal state", "WeChat does not expose per-operator NPU placement"],
      };
      this.setData({
        running: false,
        hasResult: true,
        statusText: "Complete",
        progressPercent: 100,
        summary,
        deviceText,
        engineText: `WeChat native ONNX · ${loaded.environment.version}`,
      });
    } catch (error) {
      this.setData({ running: false, errorText: error.message || "Benchmark failed" });
    } finally {
      wx.setKeepScreenOn({ keepScreenOn: false });
    }
  },

  async exportReport() {
    if (!this.report) return;
    try {
      await shareReport("c3-benchmark", this.report);
      wx.showToast({ title: "报告已导出", icon: "success" });
    } catch (error) {
      wx.showModal({ title: "导出失败", content: error.errMsg || error.message, showCancel: false });
    }
  },
});
