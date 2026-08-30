export type ModelSpec = {
  id: string;
  dataset: 'NEU-DET' | 'DeepPCB';
  method: 'V-PEFT';
  title: string;
  shortTitle: string;
  file: string;
  url: string;
  sizeBytes: number;
  sha256: string;
  inputSize: number;
  inputName: string;
  outputName: string;
  labels: string[];
  recommendedConfidence: number;
  verifiedUse: string;
  captureHint: string;
  outOfScope: string;
};

export const MODEL_CATALOG: ModelSpec[] = [
  {
    id: 'neu-vpeft-640',
    dataset: 'NEU-DET',
    method: 'V-PEFT',
    title: 'NEU-DET · V-PEFT',
    shortTitle: 'NEU-DET',
    file: 'neu_vpeft_640.onnx',
    url: '/models/neu_vpeft_640.onnx',
    sizeBytes: 10_568_717,
    sha256: '09fbeaaa79c17a0146f030945274f37774696e1a2dc61c9c18ae80e0f68c065d',
    inputSize: 640,
    inputName: 'images',
    outputName: 'output0',
    labels: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'],
    recommendedConfidence: 0.325,
    verifiedUse: 'Flat rolled-steel surface close-ups',
    captureHint: 'Fill the frame with one flat steel surface, keep the camera parallel, and use even diffuse light.',
    outOfScope: 'everyday metal objects, painted or curved parts, people, rooms, and general object recognition',
  },
  {
    id: 'deeppcb-vpeft-640',
    dataset: 'DeepPCB',
    method: 'V-PEFT',
    title: 'DeepPCB · V-PEFT',
    shortTitle: 'DeepPCB',
    file: 'deeppcb_vpeft_640.onnx',
    url: '/models/deeppcb_vpeft_640.onnx',
    sizeBytes: 10_568_692,
    sha256: '4d75d75a44e444c3b7912986ee4834e80f51ef6fcd535cd31d628827ac1edac1',
    inputSize: 640,
    inputName: 'images',
    outputName: 'output0',
    labels: ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole'],
    recommendedConfidence: 0.441,
    verifiedUse: 'DeepPCB-style bare circuit-board inspection images',
    captureHint: 'Use a sharp, top-down PCB image with traces filling the frame and minimal glare or background.',
    outOfScope: 'assembled electronics, household photos, people, rooms, and non-PCB surfaces',
  },
];

export function modelById(id: string): ModelSpec {
  return MODEL_CATALOG.find((model) => model.id === id) ?? MODEL_CATALOG[0];
}

export function formatBytes(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
