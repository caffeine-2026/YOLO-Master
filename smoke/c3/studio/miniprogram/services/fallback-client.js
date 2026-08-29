const { getRuntimeSettings } = require("./model-store");

function inferPhotoOnServer(filePath, model, confidence, iou) {
  const settings = getRuntimeSettings();
  if (!settings.fallbackApiBaseUrl) {
    return Promise.reject(new Error("Edge inference failed and no HTTPS fallback API is configured"));
  }
  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: `${settings.fallbackApiBaseUrl}/v1/infer`,
      filePath,
      name: "file",
      formData: {
        dataset: model.dataset,
        method: model.method,
        confidence: String(confidence),
        iou: String(iou),
      },
      success(result) {
        if (result.statusCode !== 200) {
          reject(new Error(`Fallback inference failed with HTTP ${result.statusCode}`));
          return;
        }
        try {
          const payload = JSON.parse(result.data);
          payload.detections = (payload.detections || []).map((detection) => ({
            ...detection,
            classId: Math.max(0, model.labels.indexOf(detection.label)),
          }));
          resolve(payload);
        } catch (error) {
          reject(new Error("Fallback API returned invalid JSON"));
        }
      },
      fail: reject,
    });
  });
}

module.exports = { inferPhotoOnServer };
