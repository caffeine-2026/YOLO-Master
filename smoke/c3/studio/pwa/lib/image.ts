import type { Detection, LetterboxTransform } from '@/lib/postprocess';

const PALETTE = ['#38bdf8', '#fb7185', '#f59e0b', '#a78bfa', '#34d399', '#22d3ee'];

export type PreparedInput = {
  data: Float32Array;
  transform: LetterboxTransform;
};

export function prepareCanvasSource(
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  inputSize: number,
  canvas: HTMLCanvasElement,
): PreparedInput {
  if (!sourceWidth || !sourceHeight) throw new Error('The source image has no dimensions.');
  canvas.width = inputSize;
  canvas.height = inputSize;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas 2D is unavailable.');
  const scale = Math.min(inputSize / sourceWidth, inputSize / sourceHeight);
  const resizedWidth = Math.max(1, Math.round(sourceWidth * scale));
  const resizedHeight = Math.max(1, Math.round(sourceHeight * scale));
  const left = Math.floor((inputSize - resizedWidth) / 2);
  const top = Math.floor((inputSize - resizedHeight) / 2);
  context.fillStyle = 'rgb(114 114 114)';
  context.fillRect(0, 0, inputSize, inputSize);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(source, 0, 0, sourceWidth, sourceHeight, left, top, resizedWidth, resizedHeight);
  const rgba = context.getImageData(0, 0, inputSize, inputSize).data;
  const plane = inputSize * inputSize;
  const tensor = new Float32Array(plane * 3);
  for (let pixel = 0, rgbaOffset = 0; pixel < plane; pixel += 1, rgbaOffset += 4) {
    tensor[pixel] = rgba[rgbaOffset] / 255;
    tensor[plane + pixel] = rgba[rgbaOffset + 1] / 255;
    tensor[plane * 2 + pixel] = rgba[rgbaOffset + 2] / 255;
  }
  return {
    data: tensor,
    transform: { inputSize, sourceWidth, sourceHeight, scale, left, top },
  };
}

export function drawDetections(
  canvas: HTMLCanvasElement,
  detections: Detection[],
  sourceWidth: number,
  sourceHeight: number,
  fit: 'cover' | 'contain' = 'cover',
): void {
  const bounds = canvas.getBoundingClientRect();
  const ratio = Math.max(1, window.devicePixelRatio || 1);
  const physicalWidth = Math.max(1, Math.round(bounds.width * ratio));
  const physicalHeight = Math.max(1, Math.round(bounds.height * ratio));
  if (canvas.width !== physicalWidth || canvas.height !== physicalHeight) {
    canvas.width = physicalWidth;
    canvas.height = physicalHeight;
  }
  const context = canvas.getContext('2d');
  if (!context) return;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, bounds.width, bounds.height);
  if (!sourceWidth || !sourceHeight) return;
  const scale = fit === 'contain'
    ? Math.min(bounds.width / sourceWidth, bounds.height / sourceHeight)
    : Math.max(bounds.width / sourceWidth, bounds.height / sourceHeight);
  const offsetX = (bounds.width - sourceWidth * scale) / 2;
  const offsetY = (bounds.height - sourceHeight * scale) / 2;
  context.lineWidth = 2;
  context.font = '600 12px ui-sans-serif, system-ui, sans-serif';
  context.textBaseline = 'top';

  for (const detection of detections) {
    const color = PALETTE[detection.classId % PALETTE.length];
    const x = offsetX + detection.x1 * scale;
    const y = offsetY + detection.y1 * scale;
    const width = Math.max(1, (detection.x2 - detection.x1) * scale);
    const height = Math.max(1, (detection.y2 - detection.y1) * scale);
    const label = `${detection.label} ${Math.round(detection.confidence * 100)}%`;
    const textWidth = context.measureText(label).width + 12;
    const labelY = Math.max(0, y - 22);
    context.strokeStyle = color;
    context.shadowColor = 'rgb(0 0 0 / 45%)';
    context.shadowBlur = 4;
    context.strokeRect(x, y, width, height);
    context.shadowBlur = 0;
    context.fillStyle = color;
    context.fillRect(x, labelY, textWidth, 20);
    context.fillStyle = '#03111c';
    context.fillText(label, x + 6, labelY + 3);
  }
}
