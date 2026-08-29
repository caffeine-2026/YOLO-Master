const SETTINGS_KEY = "c3-edge-runtime-settings-v1";
const MODEL_DIRECTORY = `${wx.env.USER_DATA_PATH}/c3-edge-models`;

function normalizeBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function getRuntimeSettings() {
  const stored = wx.getStorageSync(SETTINGS_KEY) || {};
  return {
    modelBaseUrl: normalizeBaseUrl(stored.modelBaseUrl),
    fallbackApiBaseUrl: normalizeBaseUrl(stored.fallbackApiBaseUrl),
    confidence: Number.isFinite(Number(stored.confidence)) ? Number(stored.confidence) : 0.25,
    iou: Number.isFinite(Number(stored.iou)) ? Number(stored.iou) : 0.45,
  };
}

function saveRuntimeSettings(settings) {
  const normalized = {
    ...getRuntimeSettings(),
    ...settings,
    modelBaseUrl: normalizeBaseUrl(settings.modelBaseUrl),
    fallbackApiBaseUrl: normalizeBaseUrl(settings.fallbackApiBaseUrl),
  };
  wx.setStorageSync(SETTINGS_KEY, normalized);
  const app = getApp && getApp();
  if (app && app.globalData) app.globalData.settings = normalized;
  return normalized;
}

function ensureModelDirectory() {
  const fileSystem = wx.getFileSystemManager();
  try {
    fileSystem.accessSync(MODEL_DIRECTORY);
  } catch (error) {
    fileSystem.mkdirSync(MODEL_DIRECTORY, true);
  }
}

function localModelPath(model) {
  ensureModelDirectory();
  return `${MODEL_DIRECTORY}/${model.file}`;
}

function modelIsCached(model) {
  try {
    wx.getFileSystemManager().accessSync(localModelPath(model));
    return true;
  } catch (error) {
    return false;
  }
}

function digestFile(filePath) {
  return new Promise((resolve, reject) => {
    wx.getFileSystemManager().getFileInfo({
      filePath,
      digestAlgorithm: "sha256",
      success: (result) => resolve(result.digest || ""),
      fail: reject,
    });
  });
}

function removeCachedModel(model) {
  return new Promise((resolve, reject) => {
    const path = localModelPath(model);
    wx.getFileSystemManager().unlink({
      filePath: path,
      success: resolve,
      fail(error) {
        if (String(error.errMsg || "").includes("no such file")) resolve();
        else reject(error);
      },
    });
  });
}

function downloadModel(model, onProgress = () => {}) {
  const settings = getRuntimeSettings();
  if (!settings.modelBaseUrl) {
    return Promise.reject(new Error("Set an HTTPS model base URL on the Models page first"));
  }
  ensureModelDirectory();
  const targetPath = localModelPath(model);
  const url = `${settings.modelBaseUrl}/${encodeURIComponent(model.file)}`;
  return new Promise((resolve, reject) => {
    const task = wx.downloadFile({
      url,
      success(result) {
        if (result.statusCode !== 200) {
          reject(new Error(`Model download failed with HTTP ${result.statusCode}`));
          return;
        }
        wx.getFileSystemManager().copyFile({
          srcPath: result.tempFilePath,
          destPath: targetPath,
          async success() {
            try {
              const digest = await digestFile(targetPath);
              if (model.sha256 && digest.toLowerCase() !== model.sha256.toLowerCase()) {
                await removeCachedModel(model);
                reject(new Error("Downloaded model SHA-256 does not match the catalog"));
                return;
              }
              resolve({ path: targetPath, digest });
            } catch (error) {
              reject(error);
            }
          },
          fail: reject,
        });
      },
      fail: reject,
    });
    if (task && task.onProgressUpdate) task.onProgressUpdate((event) => onProgress(event.progress || 0));
  });
}

async function ensureModel(model, onProgress) {
  const path = localModelPath(model);
  if (!modelIsCached(model)) return downloadModel(model, onProgress);
  const digest = await digestFile(path);
  if (model.sha256 && digest.toLowerCase() !== model.sha256.toLowerCase()) {
    await removeCachedModel(model);
    return downloadModel(model, onProgress);
  }
  return { path, digest };
}

module.exports = {
  MODEL_DIRECTORY,
  downloadModel,
  ensureModel,
  getRuntimeSettings,
  localModelPath,
  modelIsCached,
  removeCachedModel,
  saveRuntimeSettings,
};
