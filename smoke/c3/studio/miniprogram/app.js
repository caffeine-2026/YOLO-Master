const { MODEL_CATALOG } = require("./models/catalog");
const { getRuntimeSettings } = require("./services/model-store");

App({
  globalData: {
    models: MODEL_CATALOG,
    settings: getRuntimeSettings(),
  },

  onLaunch() {
    if (!wx.createInferenceSession) {
      console.warn("The current WeChat base library has no native inference API.");
    }
  },
});
