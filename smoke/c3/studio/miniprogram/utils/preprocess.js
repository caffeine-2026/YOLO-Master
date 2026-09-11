const PIXEL_MEAN = 114 / 255;

function letterboxRGBA(frame, inputSize) {
  const source = new Uint8Array(frame.data);
  const sourceWidth = Number(frame.width);
  const sourceHeight = Number(frame.height);
  if (!sourceWidth || !sourceHeight || source.length < sourceWidth * sourceHeight * 4) {
    throw new Error("Invalid RGBA camera frame");
  }

  const scale = Math.min(inputSize / sourceWidth, inputSize / sourceHeight);
  const resizedWidth = Math.max(1, Math.round(sourceWidth * scale));
  const resizedHeight = Math.max(1, Math.round(sourceHeight * scale));
  const left = Math.floor((inputSize - resizedWidth) / 2);
  const top = Math.floor((inputSize - resizedHeight) / 2);
  const plane = inputSize * inputSize;
  const tensor = new Float32Array(plane * 3);
  tensor.fill(PIXEL_MEAN);

  for (let outputY = 0; outputY < resizedHeight; outputY += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor(outputY / scale));
    const sourceRow = sourceY * sourceWidth * 4;
    const outputRow = (top + outputY) * inputSize + left;
    for (let outputX = 0; outputX < resizedWidth; outputX += 1) {
      const sourceX = Math.min(sourceWidth - 1, Math.floor(outputX / scale));
      const sourceIndex = sourceRow + sourceX * 4;
      const outputIndex = outputRow + outputX;
      tensor[outputIndex] = source[sourceIndex] / 255;
      tensor[plane + outputIndex] = source[sourceIndex + 1] / 255;
      tensor[plane * 2 + outputIndex] = source[sourceIndex + 2] / 255;
    }
  }

  return {
    data: tensor,
    transform: {
      inputSize,
      sourceWidth,
      sourceHeight,
      scale,
      left,
      top,
    },
  };
}

function loadImageRGBA(path, maximumSide = 1280) {
  return new Promise((resolve, reject) => {
    if (!wx.createOffscreenCanvas) {
      reject(new Error("This WeChat version cannot decode photos on-device"));
      return;
    }
    wx.getImageInfo({
      src: path,
      success(info) {
        const scale = Math.min(1, maximumSide / Math.max(info.width, info.height));
        const width = Math.max(1, Math.round(info.width * scale));
        const height = Math.max(1, Math.round(info.height * scale));
        const canvas = wx.createOffscreenCanvas({ type: "2d", width, height });
        const context = canvas.getContext("2d");
        const image = canvas.createImage();
        image.onload = () => {
          try {
            context.drawImage(image, 0, 0, width, height);
            const imageData = context.getImageData(0, 0, width, height);
            resolve({ data: imageData.data.buffer, width, height, path });
          } catch (error) {
            reject(error);
          }
        };
        image.onerror = () => reject(new Error("Unable to decode the selected image"));
        image.src = path;
      },
      fail: reject,
    });
  });
}

module.exports = { letterboxRGBA, loadImageRGBA };
