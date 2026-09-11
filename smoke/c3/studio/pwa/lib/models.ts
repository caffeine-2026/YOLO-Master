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
  inputColorMode: 'rgb' | 'grayscale';
  verifiedUse: string;
  captureHint: string;
  outOfScope: string;
  protocol: string;
  sourceCheckpoint: string;
  map5095: number;
  map50: number;
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
    recommendedConfidence: 0.25,
    inputColorMode: 'grayscale',
    verifiedUse: 'NEU-DET-style 200×200 grayscale crops of one hot-rolled steel surface',
    captureHint: 'Use the official NEU-DET test split or an equivalent tightly cropped, grayscale, front-facing steel-surface image.',
    outOfScope: 'finished metal objects, painted or curved parts, rooms, people, and general object recognition',
    protocol: 'P1 canonical 100-shot · seed 824 · 100 epochs · fixed confidence 0.25',
    sourceCheckpoint: 'neu_vpeft_seed824_e100/weights/best.pt',
    map5095: 0.328034,
    map50: 0.623852,
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
    recommendedConfidence: 0.25,
    inputColorMode: 'grayscale',
    verifiedUse: 'DeepPCB 640×640 binarized bare-board test crops',
    captureHint: 'Use the official DeepPCB tested-image split or an equivalent aligned, binarized bare-board crop.',
    outOfScope: 'assembled green PCBs, PCB drawings or schematics, phone photos, household scenes, and non-PCB surfaces',
    protocol: 'P1 canonical 100-shot · seed 824 · 100 epochs · fixed confidence 0.25',
    sourceCheckpoint: 'deeppcb_vpeft_seed824_e100/weights/best.pt',
    map5095: 0.511534,
    map50: 0.779435,
  },
];

export function modelById(id: string): ModelSpec {
  return MODEL_CATALOG.find((model) => model.id === id) ?? MODEL_CATALOG[0];
}

export function formatBytes(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
