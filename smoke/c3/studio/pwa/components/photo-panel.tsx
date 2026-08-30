'use client';

import { useEffect, useRef, useState } from 'react';
import { CheckCircle2, ChevronDown, Download, FileImage, ImagePlus, Info, LoaderCircle, ScanSearch, ShieldCheck } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { NativeSelect, NativeSelectOption } from '@/components/ui/native-select';
import type { EdgeRuntime } from '@/hooks/use-edge-runtime';
import { drawDetections } from '@/lib/image';
import type { InferenceResult } from '@/lib/inference-engine';
import { MODEL_CATALOG } from '@/lib/models';

type PhotoResult = InferenceResult & {
  id: string;
  name: string;
  url: string;
  confidenceThreshold: number;
  error?: string;
};

export function PhotoPanel({ runtime }: { runtime: EdgeRuntime }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const resultUrlsRef = useRef<string[]>([]);
  const [results, setResults] = useState<PhotoResult[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Choose industrial defect photos to run private on-device inference.');

  useEffect(() => () => resultUrlsRef.current.forEach((url) => URL.revokeObjectURL(url)), []);

  const selected = results[selectedIndex];

  function drawSelected() {
    if (!selected || !overlayRef.current || !imageRef.current) return;
    drawDetections(overlayRef.current, selected.detections, selected.width, selected.height, 'contain');
  }

  function clearResults() {
    resultUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    resultUrlsRef.current = [];
    setResults([]);
    setSelectedIndex(0);
    setProgress(0);
  }

  function changeModel(id: string) {
    const nextModel = MODEL_CATALOG.find((model) => model.id === id) ?? MODEL_CATALOG[0];
    clearResults();
    setStatus(`Use only ${nextModel.verifiedUse.toLowerCase()}.`);
    runtime.selectModel(id);
  }

  async function processFiles(files: FileList | null) {
    if (!files?.length) return;
    clearResults();
    setProcessing(true);
    setProgress(0);
    setStatus('Preparing verified model');
    try {
      await runtime.loadModel();
      const confidenceThreshold = runtime.model.recommendedConfidence;
      const selectedFiles = Array.from(files).slice(0, 9);
      const nextResults: PhotoResult[] = [];
      for (let index = 0; index < selectedFiles.length; index += 1) {
        const file = selectedFiles[index];
        setStatus(`Processing ${index + 1} / ${selectedFiles.length}`);
        const url = URL.createObjectURL(file);
        resultUrlsRef.current.push(url);
        try {
          const bitmap = await createImageBitmap(file);
          const inference = await runtime.engine.runSource(bitmap, bitmap.width, bitmap.height, confidenceThreshold, 0.45);
          bitmap.close();
          nextResults.push({ ...inference, id: `${file.name}-${file.lastModified}-${index}`, name: file.name, url, confidenceThreshold });
        } catch (error) {
          nextResults.push({
            id: `${file.name}-${file.lastModified}-${index}`,
            name: file.name,
            url,
            confidenceThreshold,
            width: 1,
            height: 1,
            detections: [],
            timings: { preprocessMs: 0, inferenceMs: 0, postprocessMs: 0, totalMs: 0 },
            error: error instanceof Error ? error.message : 'Photo inference failed.',
          });
        }
        setResults([...nextResults]);
        setProgress(Math.round(((index + 1) / selectedFiles.length) * 100));
      }
      const candidateCount = nextResults.reduce((sum, result) => sum + result.detections.length, 0);
      setStatus(candidateCount
        ? `Complete · ${candidateCount} defect candidate${candidateCount === 1 ? '' : 's'}`
        : 'Complete · no defect candidates');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Unable to prepare the model.');
    } finally {
      setProcessing(false);
    }
  }

  async function exportReport() {
    if (!results.length) return;
    const report = {
      schema: 'c3-edge-photo-report/v1',
      createdAt: new Date().toISOString(),
      model: runtime.model,
      backend: runtime.backend,
      interpretation: 'Detections are review candidates, not confirmed defects.',
      confidenceThreshold: runtime.model.recommendedConfidence,
      results: results.map(({ url: _url, ...result }) => result),
    };
    const file = new File([JSON.stringify(report, null, 2)], `c3-photo-${Date.now()}.json`, { type: 'application/json' });
    if (navigator.share && navigator.canShare?.({ files: [file] })) {
      await navigator.share({ files: [file], title: 'C3 Edge Lab photo report' });
      return;
    }
    const url = URL.createObjectURL(file);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = file.name;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const successful = results.filter((result) => !result.error);
  const totalDetections = successful.reduce((sum, result) => sum + result.detections.length, 0);
  const meanLatency = successful.length
    ? successful.reduce((sum, result) => sum + result.timings.totalMs, 0) / successful.length
    : 0;

  return (
    <section className="grid flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
      <Card className="min-h-[65svh] border-white/10 bg-card/60 ring-0">
        <CardHeader className="border-b border-white/8 pb-4 sm:grid-cols-[1fr_auto]">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-cyan-300">Photo & batch</p>
            <CardTitle className="mt-1 text-xl">Industrial defect inspection</CardTitle>
          </div>
          <NativeSelect value={runtime.selectedId} onChange={(event) => changeModel(event.target.value)}>
            {MODEL_CATALOG.map((model) => <NativeSelectOption key={model.id} value={model.id}>{model.title}</NativeSelectOption>)}
          </NativeSelect>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 p-4 sm:p-5">
          <details className="group rounded-xl border border-white/8 bg-white/[0.025] p-3 text-xs">
            <summary className="flex cursor-pointer list-none items-center gap-2 text-slate-300 marker:hidden">
              <Info className="size-4 shrink-0 text-cyan-300" />
              <span className="truncate">Model scope: {runtime.model.verifiedUse}</span>
              <ChevronDown className="ml-auto size-4 shrink-0 text-slate-500 transition group-open:rotate-180" />
            </summary>
            <p className="mt-3 border-t border-white/6 pt-3 leading-5 text-slate-500">{runtime.model.captureHint} Not designed for {runtime.model.outOfScope}.</p>
          </details>
          <div className="relative grid min-h-[42svh] place-items-center overflow-hidden rounded-[22px] border border-dashed border-white/12 bg-[#050b11]">
            {selected ? (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img ref={imageRef} src={selected.url} alt={selected.name} onLoad={drawSelected} className={`absolute inset-0 h-full w-full object-contain ${runtime.model.inputColorMode === 'grayscale' ? 'grayscale' : ''}`} />
                <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />
                {selected.error && <div className="absolute inset-x-4 bottom-4 rounded-xl bg-rose-950/85 p-3 text-sm text-rose-200">{selected.error}</div>}
              </>
            ) : (
              <div className="max-w-sm px-6 text-center">
                <div className="mx-auto mb-4 grid size-16 place-items-center rounded-2xl bg-cyan-300/8"><ImagePlus className="size-7 text-cyan-300" /></div>
                <p className="font-medium">Drop in up to 9 photos</p>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">Nothing is uploaded. Images and model inference remain in this browser.</p>
              </div>
            )}
          </div>

          {selected && !selected.error && (
            <div className={`flex items-center gap-2 rounded-xl border px-3 py-2.5 text-sm ${selected.detections.length ? 'border-rose-300/15 bg-rose-300/[0.045] text-rose-100' : 'border-emerald-300/15 bg-emerald-300/[0.045] text-emerald-100'}`}>
              {selected.detections.length ? <ScanSearch className="size-4 shrink-0 text-rose-300" /> : <CheckCircle2 className="size-4 shrink-0 text-emerald-300" />}
              <strong>{selected.detections.length ? `${selected.detections.length} defect candidate${selected.detections.length === 1 ? '' : 's'}` : 'No defect candidates'}</strong>
            </div>
          )}

          <input ref={inputRef} className="hidden" type="file" accept="image/*" multiple onChange={(event) => void processFiles(event.target.files)} />
          <div className="flex flex-wrap gap-2">
            <Button type="button" onClick={() => inputRef.current?.click()} disabled={processing} className="h-10 bg-cyan-300 px-4 text-cyan-950 hover:bg-cyan-200">
              {processing ? <LoaderCircle className="animate-spin" /> : <FileImage />} Choose photos
            </Button>
            <Button type="button" variant="outline" onClick={() => void exportReport()} disabled={!results.length || processing} className="h-10 border-white/10">
              <Download /> Export JSON
            </Button>
            <Badge variant="outline" className="ml-auto h-8 border-emerald-300/15 bg-emerald-300/8 text-emerald-300"><ShieldCheck /> On-device only</Badge>
          </div>

          {(processing || results.length > 0) && (
            <div>
              <div className="mb-2 flex justify-between text-xs text-muted-foreground"><span>{status}</span><span>{progress}%</span></div>
              <div className="h-1.5 overflow-hidden rounded-full bg-white/8"><div className="h-full bg-cyan-300 transition-all" style={{ width: `${progress}%` }} /></div>
            </div>
          )}
        </CardContent>
      </Card>

      <aside className="flex min-h-0 flex-col gap-3">
        <div className="grid grid-cols-3 gap-2">
          {[
            ['Images', results.length],
            ['Candidates', totalDetections],
            ['End-to-end ms', meanLatency.toFixed(1)],
          ].map(([label, value]) => (
            <Card key={String(label)} className="border-white/10 bg-card/60 ring-0"><CardContent className="p-3 text-center"><strong className="block font-mono text-lg text-cyan-200">{value}</strong><span className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</span></CardContent></Card>
          ))}
        </div>
        <Card className="flex-1 border-white/10 bg-card/60 ring-0">
          <CardHeader><CardTitle className="flex items-center gap-2"><ScanSearch className="size-4 text-violet-300" /> Batch results</CardTitle></CardHeader>
          <CardContent className="space-y-2 px-3 pb-3">
            {results.length === 0 ? (
              <p className="px-2 py-10 text-center text-sm text-muted-foreground">Completed images will appear here.</p>
            ) : results.map((result, index) => (
              <button
                type="button"
                key={result.id}
                onClick={() => setSelectedIndex(index)}
                className={`flex w-full items-center justify-between rounded-xl border px-3 py-3 text-left transition ${selectedIndex === index ? 'border-cyan-300/25 bg-cyan-300/8' : 'border-white/6 bg-white/[0.025] hover:bg-white/[0.05]'}`}
              >
                <span className="min-w-0"><span className="block truncate text-sm">{result.name}</span><span className="text-xs text-muted-foreground">{result.error ? 'Failed' : `${result.detections.length} review candidates`}</span></span>
                <span className="ml-3 font-mono text-xs text-slate-400">{result.timings.totalMs.toFixed(1)} ms</span>
              </button>
            ))}
          </CardContent>
        </Card>
      </aside>
    </section>
  );
}
