const { EdgeInferenceEngine, inferenceEnvironment } = require("../../services/inference-engine");
const { getRuntimeSettings } = require("../../services/model-store");
const { drawDetections, prepareCanvas } = require("../../utils/overlay");
const { nowMs } = require("../../utils/time");

Page({
  data: {
    models: [],
    modelTitles: [],
    selectedIndex: 0,
    running: false,
    loading: false,
    engineReady: false,
    backendText: "Checking engine",
    downloadProgress: 0,
    errorText: "",
    cameraPosition: "back",
    flashMode: "off",
    confidencePercent: 25,
    confidenceText: "0.25",
    fps: "0.0",
    preprocessMs: "0.0",
    inferenceMs: "0.0",
    postprocessMs: "0.0",
    totalMs: "0.0",
    detectionCount: 0,
  },

  onLoad() {
    const app = getApp();
    const settings = getRuntimeSettings();
    this.engine = new EdgeInferenceEngine();
    this.inFlight = false;
    this.lastCompleted = 0;
    this.lastUiUpdate = 0;
    this.setData({
      models: app.globalData.models,
      modelTitles: app.globalData.models.map((model) => model.title),
      confidencePercent: Math.round(settings.confidence * 100),
      confidenceText: settings.confidence.toFixed(2),
    });
    inferenceEnvironment().then((environment) => {
      this.setData({ backendText: environment.supported ? `Native ONNX · ${environment.version}` : "Edge engine unavailable" });
    });
  },

  async onReady() {
    try {
      this.overlay = await prepareCanvas(this, "#liveOverlay");
    } catch (error) {
      this.setData({ errorText: error.message });
    }
    this.camera = wx.createCameraContext(this);
  },

  onHide() {
    this.stopFrames();
  },

  onUnload() {
    this.stopFrames();
    if (this.engine) this.engine.destroy();
  },

  currentModel() {
    return this.data.models[this.data.selectedIndex];
  },

  async prepareEngine() {
    const model = this.currentModel();
    if (this.engine.model && this.engine.model.id === model.id) return;
    this.setData({ loading: true, engineReady: false, errorText: "", downloadProgress: 0 });
    try {
      const loaded = await this.engine.load(model, (progress) => this.setData({ downloadProgress: progress }));
      this.setData({
        loading: false,
        engineReady: true,
        backendText: `Native ONNX · ${loaded.environment.version} · NPU requested`,
        downloadProgress: 100,
      });
    } catch (error) {
      this.setData({ loading: false, engineReady: false, errorText: error.message });
      throw error;
    }
  },

  async toggleRun() {
    if (this.data.running) {
      this.stopFrames();
      return;
    }
    try {
      await this.prepareEngine();
      this.startFrames();
    } catch (error) {
      // prepareEngine already surfaced the actionable message.
    }
  },

  startFrames() {
    if (!this.camera || !this.camera.onCameraFrame) {
      this.setData({ errorText: "Real-time camera frames are unavailable in this WeChat version" });
      return;
    }
    this.frameListener = this.camera.onCameraFrame((frame) => this.processFrame(frame));
    this.frameListener.start();
    this.lastCompleted = 0;
    this.setData({ running: true, errorText: "" });
  },

  stopFrames() {
    if (this.frameListener) {
      try {
        this.frameListener.stop();
      } catch (error) {
        console.warn("Unable to stop camera frame listener", error);
      }
    }
    this.frameListener = null;
    this.inFlight = false;
    if (this.data.running) this.setData({ running: false });
  },

  async processFrame(frame) {
    if (!this.data.running || this.inFlight) return;
    this.inFlight = true;
    try {
      const settings = getRuntimeSettings();
      const confidence = this.data.confidencePercent / 100;
      const result = await this.engine.runRGBA(frame, { confidence, iou: settings.iou });
      drawDetections(this.overlay, result.detections, frame.width, frame.height, "cover");
      const completed = nowMs();
      const observedFps = this.lastCompleted ? 1000 / Math.max(1, completed - this.lastCompleted) : 1000 / Math.max(1, result.timings.totalMs);
      this.lastCompleted = completed;
      if (completed - this.lastUiUpdate >= 200) {
        this.lastUiUpdate = completed;
        this.setData({
          fps: observedFps.toFixed(1),
          preprocessMs: result.timings.preprocessMs.toFixed(1),
          inferenceMs: result.timings.inferenceMs.toFixed(1),
          postprocessMs: result.timings.postprocessMs.toFixed(1),
          totalMs: result.timings.totalMs.toFixed(1),
          detectionCount: result.detections.length,
        });
      }
    } catch (error) {
      this.stopFrames();
      this.setData({ errorText: error.message || "Live inference failed" });
    } finally {
      this.inFlight = false;
    }
  },

  async onModelChange(event) {
    const selectedIndex = Number(event.detail.value);
    const shouldResume = this.data.running;
    this.stopFrames();
    this.engine.destroy();
    this.setData({ selectedIndex, engineReady: false, detectionCount: 0 });
    if (this.overlay) drawDetections(this.overlay, [], 1, 1);
    if (shouldResume) await this.toggleRun();
  },

  onConfidenceChange(event) {
    const confidencePercent = Number(event.detail.value);
    this.setData({ confidencePercent, confidenceText: (confidencePercent / 100).toFixed(2) });
  },

  toggleFlash() {
    this.setData({ flashMode: this.data.flashMode === "torch" ? "off" : "torch" });
  },

  switchCamera() {
    this.setData({ cameraPosition: this.data.cameraPosition === "back" ? "front" : "back" });
  },

  onCameraError(event) {
    this.setData({ errorText: event.detail.errMsg || "Camera permission is required" });
  },

  openModels() {
    wx.switchTab({ url: "/pages/models/index" });
  },
});
