export type LetterboxTransform = {
  inputSize: number;
  sourceWidth: number;
  sourceHeight: number;
  scale: number;
  left: number;
  top: number;
};

export type Detection = {
  classId: number;
  label: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

export function intersectionOverUnion(a: Detection, b: Detection): number {
  const left = Math.max(a.x1, b.x1);
  const top = Math.max(a.y1, b.y1);
  const right = Math.min(a.x2, b.x2);
  const bottom = Math.min(a.y2, b.y2);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - intersection;
  return union > 0 ? intersection / union : 0;
}

export function decodeOutput(
  outputData: ArrayLike<number>,
  shape: readonly number[],
  labels: string[],
  transform: LetterboxTransform,
  confidenceThreshold = 0.25,
  iouThreshold = 0.45,
): Detection[] {
  if (shape.length !== 3) throw new Error(`Unexpected model output shape: [${shape.join(', ')}]`);
  const channelFirst = shape[1] <= 128;
  const channels = channelFirst ? shape[1] : shape[2];
  const candidates = channelFirst ? shape[2] : shape[1];
  const classCount = channels - 4;
  if (classCount !== labels.length) {
    throw new Error(`Model output has ${classCount} classes but the catalog has ${labels.length}.`);
  }

  const valueAt = channelFirst
    ? (candidate: number, channel: number) => Number(outputData[channel * candidates + candidate])
    : (candidate: number, channel: number) => Number(outputData[candidate * channels + channel]);
  const boxes: Detection[] = [];

  for (let candidate = 0; candidate < candidates; candidate += 1) {
    let bestClass = 0;
    let bestScore = valueAt(candidate, 4);
    for (let classIndex = 1; classIndex < classCount; classIndex += 1) {
      const score = valueAt(candidate, 4 + classIndex);
      if (score > bestScore) {
        bestScore = score;
        bestClass = classIndex;
      }
    }
    if (bestScore < confidenceThreshold) continue;

    const centerX = valueAt(candidate, 0);
    const centerY = valueAt(candidate, 1);
    const width = valueAt(candidate, 2);
    const height = valueAt(candidate, 3);
    boxes.push({
      classId: bestClass,
      label: labels[bestClass],
      confidence: bestScore,
      x1: clamp((centerX - width / 2 - transform.left) / transform.scale, 0, transform.sourceWidth),
      y1: clamp((centerY - height / 2 - transform.top) / transform.scale, 0, transform.sourceHeight),
      x2: clamp((centerX + width / 2 - transform.left) / transform.scale, 0, transform.sourceWidth),
      y2: clamp((centerY + height / 2 - transform.top) / transform.scale, 0, transform.sourceHeight),
    });
  }

  boxes.sort((a, b) => b.confidence - a.confidence);
  const selected: Detection[] = [];
  for (const box of boxes.slice(0, 300)) {
    const suppressed = selected.some(
      (picked) => picked.classId === box.classId && intersectionOverUnion(picked, box) > iouThreshold,
    );
    if (!suppressed) selected.push(box);
    if (selected.length >= 100) break;
  }
  return selected;
}
