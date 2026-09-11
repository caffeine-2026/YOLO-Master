const { EdgeInferenceEngine } = require("../../services/inference-engine");
const { inferPhotoOnServer } = require("../../services/fallback-client");
const { getRuntimeSettings } = require("../../services/model-store");
const { drawDetections, prepareCanvas } = require("../../utils/overlay");
const { loadImageRGBA } = require("../../utils/preprocess");
const { shareReport } = require("../../utils/report");

Page({
  data: {
    models: [],
    modelTitles: [],
    selectedIndex: 0,
    previewPath: "",
    processing: false,
    processed: 0,
    total: 0,
    progressPercent: 0,
    statusText: "",
    errorText: "",
    results: [],
    totalDetections: 0,
    meanLatency: "0.0",
  },

  onLoad() {
    const models = getApp().globalData.models;
    this.engine = new EdgeInferenceEngine();
    this.sourceFrames = new Map();
    this.setData({ models, modelTitles: models.map((model) => model.title) });
  },

  async onReady() {
    try {
      this.overlay = await prepareCanvas(this, "#photoOverlay");
    } catch (error) {
      // The canvas becomes available after an image is selected; it is prepared lazily there as well.
    }
  },

  onUnload() {
    if (this.engine) this.engine.destroy();
  },

  currentModel() {
    return this.data.models[this.data.selectedIndex];
  },

  onModelChange(event) {
    this.engine.destroy();
    this.setData({ selectedIndex: Number(event.detail.value), results: [], previewPath: "", errorText: "" });
  },

  chooseImages() {
    wx.chooseMedia({
      count: 9,
      mediaType: ["image"],
      sourceType: ["album", "camera"],
      success: (result) => this.processFiles(result.tempFiles.map((item) => item.tempFilePath)),
      fail: (error) => {
        if (!String(error.errMsg || "").includes("cancel")) this.setData({ errorText: error.errMsg });
      },
    });
  },

  async ensureOverlay() {
    if (!this.overlay) this.overlay = await prepareCanvas(this, "#photoOverlay");
  },

  async processFiles(paths) {
    if (!paths.length) return;
    const model = this.currentModel();
    const settings = getRuntimeSettings();
    const results = [];
    let edgeReady = false;
    this.setData({
      previewPath: paths[0],
      processing: true,
      processed: 0,
      total: paths.length,
      progressPercent: 0,
      statusText: "Preparing model",
      errorText: "",
      results: [],
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    await this.ensureOverlay();

    try {
      await this.engine.load(model, (progress) => this.setData({ statusText: `Downloading model ${progress}%` }));
      edgeReady = true;
    } catch (error) {
      if (!settings.fallbackApiBaseUrl) {
        this.setData({ processing: false, errorText: error.message });
        return;
      }
      this.setData({ statusText: "Edge unavailable · using server fallback" });
    }

    for (let index = 0; index < paths.length; index += 1) {
      const path = paths[index];
      try {
        let record;
        if (edgeReady) {
          const frame = await loadImageRGBA(path);
          const output = await this.engine.runRGBA(frame, { confidence: settings.confidence, iou: settings.iou });
          this.sourceFrames.set(path, frame);
          record = {
            id: `${Date.now()}-${index}`,
            path,
            name: path.split("/").pop(),
            backend: "WeChat native ONNX",
            detections: output.detections,
            totalMs: Number(output.timings.totalMs.toFixed(1)),
            timings: output.timings,
            width: frame.width,
            height: frame.height,
          };
        } else {
          const output = await inferPhotoOnServer(path, model, settings.confidence, settings.iou);
          record = {
            id: `${Date.now()}-${index}`,
            path,
            name: path.split("/").pop(),
            backend: "Server fallback",
            detections: output.detections,
            totalMs: Number(output.latency_ms.toFixed(1)),
            timings: output.timings || {},
            width: output.width,
            height: output.height,
          };
        }
        results.push(record);
        if (index === 0) drawDetections(this.overlay, record.detections, record.width, record.height, "contain");
      } catch (error) {
        results.push({
          id: `${Date.now()}-${index}`,
          path,
          name: path.split("/").pop(),
          backend: "Failed",
          detections: [],
          totalMs: 0,
          error: error.message,
        });
      }
      const processed = index + 1;
      this.setData({
        processed,
        progressPercent: Math.round((processed / paths.length) * 100),
        statusText: `Processing ${processed} / ${paths.length}`,
      });
    }

    const successful = results.filter((item) => !item.error);
    const totalDetections = successful.reduce((sum, item) => sum + item.detections.length, 0);
    const meanLatency = successful.length
      ? successful.reduce((sum, item) => sum + item.totalMs, 0) / successful.length
      : 0;
    this.setData({
      processing: false,
      results,
      totalDetections,
      meanLatency: meanLatency.toFixed(1),
      statusText: "Complete",
    });
  },

  async showResult(event) {
    const result = this.data.results[Number(event.currentTarget.dataset.index)];
    if (!result || result.error) return;
    this.setData({ previewPath: result.path });
    await new Promise((resolve) => setTimeout(resolve, 30));
    await this.ensureOverlay();
    drawDetections(this.overlay, result.detections, result.width, result.height, "contain");
  },

  async exportReport() {
    try {
      const payload = {
        schema: "c3-edge-photo-report/v1",
        createdAt: new Date().toISOString(),
        model: this.currentModel(),
        results: this.data.results.map(({ path, ...result }) => result),
      };
      await shareReport("c3-photo", payload);
      wx.showToast({ title: "报告已导出", icon: "success" });
    } catch (error) {
      wx.showModal({ title: "导出失败", content: error.errMsg || error.message, showCancel: false });
    }
  },
});
