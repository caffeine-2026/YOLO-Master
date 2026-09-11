function safeTimestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function writeJsonReport(prefix, payload) {
  return new Promise((resolve, reject) => {
    const path = `${wx.env.USER_DATA_PATH}/${prefix}-${safeTimestamp()}.json`;
    wx.getFileSystemManager().writeFile({
      filePath: path,
      data: `${JSON.stringify(payload, null, 2)}\n`,
      encoding: "utf8",
      success: () => resolve(path),
      fail: reject,
    });
  });
}

async function shareReport(prefix, payload) {
  const path = await writeJsonReport(prefix, payload);
  if (wx.shareFileMessage) {
    await new Promise((resolve, reject) => {
      wx.shareFileMessage({ filePath: path, fileName: path.split("/").pop(), success: resolve, fail: reject });
    });
  } else {
    wx.setClipboardData({ data: JSON.stringify(payload, null, 2) });
  }
  return path;
}

module.exports = { shareReport, writeJsonReport };
