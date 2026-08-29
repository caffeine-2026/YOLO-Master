'use client';

import { useRef, useState } from 'react';
import { CircleStop, Cpu, Download, Gauge, Play, ShieldAlert, TimerReset } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { NativeSelect, NativeSelectOption } from '@/components/ui/native-select';
import type { EdgeRuntime } from '@/hooks/use-edge-runtime';
import { summarizeLatencies, type LatencySummary } from '@/lib/metrics';
import { MODEL_CATALOG } from '@/lib/models';

const EMPTY_SUMMARY: LatencySummary = { count: 0, mean: 0, p50: 0, p95: 0, min: 0, max: 0, fps: 0 };

export function BenchPanel({ runtime }: { runtime: EdgeRuntime }) {
  const cancelRef = useRef(false);
  const reportRef = useRef<Record<string, unknown> | null>(null);
  const [iterations, setIterations] = useState(30);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState('Ready to benchmark');
  const [progress, setProgress] = useState(0);
  const [summary, setSummary] = useState<LatencySummary>(EMPTY_SUMMARY);
  const [error, setError] = useState('');
  const [hasReport, setHasReport] = useState(false);

  async function runBenchmark() {
    cancelRef.current = false;
    setRunning(true);
    setError('');
    setProgress(0);
    setSummary(EMPTY_SUMMARY);
    setHasReport(false);
    try {
      setStatus('Preparing verified model');
      await runtime.loadModel();
      const input = new Float32Array(3 * runtime.model.inputSize * runtime.model.inputSize);
      input.fill(114 / 255);
      for (let index = 0; index < 5; index += 1) {
        if (cancelRef.current) throw new Error('Benchmark cancelled.');
        setStatus(`Warmup ${index + 1} / 5`);
        await runtime.engine.runTensor(input);
      }
      const latencies: number[] = [];
      for (let index = 0; index < iterations; index += 1) {
        if (cancelRef.current) throw new Error('Benchmark cancelled.');
        const result = await runtime.engine.runTensor(input);
        latencies.push(result.inferenceMs);
        if ((index + 1) % 3 === 0 || index + 1 === iterations) {
          setStatus(`Measured ${index + 1} / ${iterations}`);
          setProgress(Math.round(((index + 1) / iterations) * 100));
        }
      }
      const nextSummary = summarizeLatencies(latencies);
      setSummary(nextSummary);
      setStatus('Benchmark complete');
      reportRef.current = {
        schema: 'c3-edge-benchmark/v1',
        createdAt: new Date().toISOString(),
        model: runtime.model,
        backend: runtime.backend,
        configuration: { warmup: 5, iterations, inputShape: [1, 3, 640, 640] },
        browser: { userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency },
        latenciesMs: latencies,
        summary: nextSummary,
        limitations: ['Browser APIs do not expose thermal state.', 'Execution-provider operator placement is not exposed.'],
      };
      setHasReport(true);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : 'Benchmark failed.';
      setError(message);
      setStatus(message);
    } finally {
      setRunning(false);
    }
  }

  function exportReport() {
    if (!reportRef.current) return;
    const blob = new Blob([JSON.stringify(reportRef.current, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `c3-benchmark-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="grid flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
      <Card className="min-h-[65svh] border-white/10 bg-card/60 ring-0">
        <CardHeader className="border-b border-white/8 pb-4 sm:grid-cols-[1fr_auto]">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-violet-300">Model-only benchmark</p>
            <CardTitle className="mt-1 text-xl">Sustained edge latency</CardTitle>
          </div>
          <NativeSelect value={runtime.selectedId} onChange={(event) => runtime.selectModel(event.target.value)} disabled={running}>
            {MODEL_CATALOG.map((model) => <NativeSelectOption key={model.id} value={model.id}>{model.title}</NativeSelectOption>)}
          </NativeSelect>
        </CardHeader>
        <CardContent className="flex flex-1 flex-col p-4 sm:p-6">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {[
              ['Mean', summary.mean, 'ms'],
              ['P50', summary.p50, 'ms'],
              ['P95', summary.p95, 'ms'],
              ['Throughput', summary.fps, 'FPS'],
            ].map(([label, value, unit], index) => (
              <div key={String(label)} className={`rounded-[20px] border p-4 ${index === 3 ? 'border-cyan-300/18 bg-cyan-300/7' : 'border-white/8 bg-white/[0.025]'}`}>
                <span className="text-xs uppercase tracking-[0.12em] text-muted-foreground">{label}</span>
                <div className="mt-3 flex items-end gap-2"><strong className="font-mono text-3xl tracking-tight">{Number(value).toFixed(1)}</strong><span className="pb-1 text-xs text-muted-foreground">{unit}</span></div>
              </div>
            ))}
          </div>

          <div className="my-5 grid flex-1 place-items-center rounded-[24px] border border-white/8 bg-[#071019]/55 p-5">
            <div className="w-full max-w-lg text-center">
              <div className="relative mx-auto mb-5 grid size-36 place-items-center rounded-full border border-violet-300/15 bg-violet-300/5">
                <div className="absolute inset-3 rounded-full border border-dashed border-violet-300/20" />
                <div><strong className="block font-mono text-4xl text-violet-200">{progress}%</strong><span className="text-xs text-violet-300/60">{status}</span></div>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-white/8"><div className="h-full rounded-full bg-gradient-to-r from-violet-400 to-cyan-300 transition-all duration-300" style={{ width: `${progress}%` }} /></div>
              {error && <p className="mt-4 text-sm text-rose-300">{error}</p>}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <NativeSelect value={String(iterations)} onChange={(event) => setIterations(Number(event.target.value))} disabled={running}>
              <NativeSelectOption value="30">30 iterations</NativeSelectOption>
              <NativeSelectOption value="100">100 iterations</NativeSelectOption>
            </NativeSelect>
            <Button type="button" onClick={() => void runBenchmark()} disabled={running} className="h-10 bg-violet-300 px-4 text-violet-950 hover:bg-violet-200"><Play /> Run benchmark</Button>
            {running && <Button type="button" variant="destructive" onClick={() => { cancelRef.current = true; }} className="h-10"><CircleStop /> Cancel</Button>}
            <Button type="button" variant="outline" onClick={exportReport} disabled={!hasReport || running} className="ml-auto h-10 border-white/10"><Download /> Export</Button>
          </div>
        </CardContent>
      </Card>

      <aside className="flex flex-col gap-3">
        <Card className="border-white/10 bg-card/60 ring-0"><CardContent className="p-4"><div className="mb-4 flex items-center gap-2 font-medium"><Cpu className="size-4 text-cyan-300" /> Runtime configuration</div><dl className="space-y-3 text-sm"><div className="flex justify-between"><dt className="text-muted-foreground">Backend</dt><dd>{runtime.backend}</dd></div><div className="flex justify-between"><dt className="text-muted-foreground">Warmup</dt><dd>5 iterations</dd></div><div className="flex justify-between"><dt className="text-muted-foreground">Measured</dt><dd>{iterations}</dd></div><div className="flex justify-between"><dt className="text-muted-foreground">Tensor</dt><dd className="font-mono">float32</dd></div></dl></CardContent></Card>
        <Card className="border-white/10 bg-card/60 ring-0"><CardContent className="p-4"><div className="mb-3 flex items-center gap-2 font-medium"><Gauge className="size-4 text-emerald-300" /> Range</div><div className="grid grid-cols-2 gap-2"><div className="rounded-xl bg-white/[0.035] p-3"><span className="text-xs text-muted-foreground">Minimum</span><strong className="mt-1 block font-mono">{summary.min.toFixed(1)} ms</strong></div><div className="rounded-xl bg-white/[0.035] p-3"><span className="text-xs text-muted-foreground">Maximum</span><strong className="mt-1 block font-mono">{summary.max.toFixed(1)} ms</strong></div></div></CardContent></Card>
        <Card className="border-amber-300/12 bg-amber-300/[0.035] ring-0"><CardContent className="p-4"><div className="mb-2 flex items-center gap-2 text-sm font-medium text-amber-200"><ShieldAlert className="size-4" /> Honest measurement boundary</div><p className="text-xs leading-5 text-amber-100/60">Browsers expose wall-clock model latency but not device temperature or per-operator CPU/GPU placement. Long runs can still reveal throttling through latency drift.</p></CardContent></Card>
        <Badge variant="outline" className="h-8 self-start border-white/10 text-muted-foreground"><TimerReset /> {summary.count} measured runs</Badge>
      </aside>
    </section>
  );
}
