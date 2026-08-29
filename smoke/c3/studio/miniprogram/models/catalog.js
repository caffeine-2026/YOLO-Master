const MODEL_CATALOG = [
  {
    id: "neu-vpeft-640",
    dataset: "NEU-DET",
    method: "V-PEFT",
    title: "NEU-DET · V-PEFT",
    file: "neu_vpeft_640.onnx",
    inputSize: 640,
    inputName: "images",
    outputName: "output0",
    labels: ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
    sha256: "09fbeaaa79c17a0146f030945274f37774696e1a2dc61c9c18ae80e0f68c065d",
    note: "100-epoch final · seed 824 · LoRA merged export",
  },
  {
    id: "deeppcb-vpeft-640",
    dataset: "DeepPCB",
    method: "V-PEFT",
    title: "DeepPCB · V-PEFT",
    file: "deeppcb_vpeft_640.onnx",
    inputSize: 640,
    inputName: "images",
    outputName: "output0",
    labels: ["open", "short", "mousebite", "spur", "copper", "pin-hole"],
    sha256: "4d75d75a44e444c3b7912986ee4834e80f51ef6fcd535cd31d628827ac1edac1",
    note: "100-epoch final · seed 824 · LoRA merged export",
  },
];

function modelById(id) {
  return MODEL_CATALOG.find((model) => model.id === id) || MODEL_CATALOG[0];
}

module.exports = { MODEL_CATALOG, modelById };
