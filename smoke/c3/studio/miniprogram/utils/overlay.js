const PALETTE = ["#38bdf8", "#fb7185", "#f59e0b", "#a78bfa", "#34d399", "#22d3ee"];

function prepareCanvas(page, selector) {
  return new Promise((resolve, reject) => {
    page
      .createSelectorQuery()
      .select(selector)
      .fields({ node: true, size: true })
      .exec((results) => {
        const result = results && results[0];
        if (!result || !result.node) {
          reject(new Error(`Canvas '${selector}' is unavailable`));
          return;
        }
        const device = wx.getDeviceInfo ? wx.getDeviceInfo() : wx.getSystemInfoSync();
        const ratio = Number(device.pixelRatio || 1);
        const canvas = result.node;
        canvas.width = Math.max(1, Math.round(result.width * ratio));
        canvas.height = Math.max(1, Math.round(result.height * ratio));
        const context = canvas.getContext("2d");
        context.scale(ratio, ratio);
        resolve({ canvas, context, width: result.width, height: result.height, ratio });
      });
  });
}

function drawDetections(surface, detections, sourceWidth, sourceHeight, fit = "cover") {
  if (!surface) return;
  const { context, width, height } = surface;
  context.clearRect(0, 0, width, height);
  if (!sourceWidth || !sourceHeight) return;
  const scale = fit === "contain" ? Math.min(width / sourceWidth, height / sourceHeight) : Math.max(width / sourceWidth, height / sourceHeight);
  const offsetX = (width - sourceWidth * scale) / 2;
  const offsetY = (height - sourceHeight * scale) / 2;
  context.textBaseline = "top";
  context.lineWidth = 2;
  context.font = "600 12px -apple-system, sans-serif";

  detections.forEach((detection) => {
    const color = PALETTE[detection.classId % PALETTE.length];
    const x = offsetX + detection.x1 * scale;
    const y = offsetY + detection.y1 * scale;
    const boxWidth = Math.max(1, (detection.x2 - detection.x1) * scale);
    const boxHeight = Math.max(1, (detection.y2 - detection.y1) * scale);
    const label = `${detection.label} ${Math.round(detection.confidence * 100)}%`;
    const textWidth = context.measureText(label).width + 12;
    const labelY = Math.max(0, y - 22);
    context.strokeStyle = color;
    context.strokeRect(x, y, boxWidth, boxHeight);
    context.fillStyle = color;
    context.fillRect(x, labelY, textWidth, 20);
    context.fillStyle = "#03111c";
    context.fillText(label, x + 6, labelY + 3);
  });
}

module.exports = { drawDetections, prepareCanvas };
