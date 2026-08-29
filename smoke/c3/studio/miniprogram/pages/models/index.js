const { EdgeInferenceEngine, inferenceEnvironment } = require("../../services/inference-engine");
const {
  downloadModel,
  getRuntimeSettings,
  modelIsCached,
  removeCachedModel,
  saveRuntimeSettings,
} = require("../../services/model-store");
const { modelById } = require("../../models/catalog");

Page({
  data: {
    environment: { supported: false, version: "checking" },
    modelBaseUrl: "",
    fallbackApiBaseUrl: "",
    confidencePercent: 25,
    confidenceText: "0.25",
    iouPercent: 45,
    iouText: "0.45",
    modelRows: [],
    activeDownloadId: "",
    downloadProgress: 0,
    message: "",
  },

  onLoad() {
    const settings = getRuntimeSettings();
    this.setData({
      modelBaseUrl: settings.modelBaseUrl,
      fallbackApiBaseUrl: settings.fallbackApiBaseUrl,
      confidencePercent: Math.round(settings.confidence * 100),
      confidenceText: settings.confidence.toFixed(2),
      iouPercent: Math.round(settings.iou * 100),
      iouText: settings.iou.toFixed(2),
    });
    inferenceEnvironment().then((environment) => this.setData({ environment }));
    this.refreshRows();
  },

  onShow() {
    this.refreshRows();
  },

  refreshRows() {
    const models = getApp().globalData.models;
    this.setData({
      modelRows: models.map((model) => ({
        ...model,
        cached: modelIsCached(model),
        shaShort: model.sha256 ? `${model.sha256.slice(0, 12)}…` : "generated at export",
      })),
    });
  },

  onModelUrlInput(event) {
    this.setData({ modelBaseUrl: event.detail.value });
  },

  onFallbackUrlInput(event) {
    this.setData({ fallbackApiBaseUrl: event.detail.value });
  },

  onConfidenceChange(event) {
    const confidencePercent = Number(event.detail.value);
    this.setData({ confidencePercent, confidenceText: (confidencePercent / 100).toFixed(2) });
  },

  onIouChange(event) {
    const iouPercent = Number(event.detail.value);
    this.setData({ iouPercent, iouText: (iouPercent / 100).toFixed(2) });
  },

  saveSettings() {
    const modelBaseUrl = this.data.modelBaseUrl.trim();
    const fallbackApiBaseUrl = this.data.fallbackApiBaseUrl.trim();
    const invalid = [modelBaseUrl, fallbackApiBaseUrl].find((value) => value && !/^https:\/\//i.test(value) && !/^http:\/\/(127\.0\.0\.1|localhost)/i.test(value));
    if (invalid) {
      wx.showModal({ title: "Invalid URL", content: "Production endpoints must use HTTPS.", showCancel: false });
      return;
    }
    saveRuntimeSettings({
      modelBaseUrl,
      fallbackApiBaseUrl,
      confidence: this.data.confidencePercent / 100,
      iou: this.data.iouPercent / 100,
    });
    wx.showToast({ title: "Saved", icon: "success" });
  },

  async download(event) {
    const id = event.currentTarget.dataset.id;
    const model = modelById(id);
    this.setData({ activeDownloadId: id, downloadProgress: 0, message: `Preparing ${model.title}` });
    try {
      const result = await downloadModel(model, (progress) => this.setData({ downloadProgress: progress }));
      this.setData({ message: `${model.title} verified · SHA-256 ${result.digest.slice(0, 16)}…` });
      this.refreshRows();
    } catch (error) {
      this.setData({ message: error.errMsg || error.message });
    } finally {
      this.setData({ activeDownloadId: "" });
    }
  },

  async remove(event) {
    const model = modelById(event.currentTarget.dataset.id);
    try {
      await removeCachedModel(model);
      this.setData({ message: `${model.title} cache deleted` });
      this.refreshRows();
    } catch (error) {
      this.setData({ message: error.errMsg || error.message });
    }
  },

  async loadTest(event) {
    const model = modelById(event.currentTarget.dataset.id);
    const engine = new EdgeInferenceEngine();
    this.setData({ message: `Compiling ${model.title}…`, activeDownloadId: model.id });
    try {
      const loaded = await engine.load(model, (progress) => this.setData({ downloadProgress: progress }));
      this.setData({ message: `${model.title} loaded successfully · engine ${loaded.environment.version}` });
      this.refreshRows();
    } catch (error) {
      this.setData({ message: error.errMsg || error.message });
    } finally {
      engine.destroy();
      this.setData({ activeDownloadId: "" });
    }
  },
});
