'use client';

import { useCallback, useEffect, useState } from 'react';
import { Box, CheckCircle2, Cpu, Database, DownloadCloud, HardDrive, LoaderCircle, RefreshCw, ShieldCheck, Trash2 } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { EdgeRuntime } from '@/hooks/use-edge-runtime';
import { ensureModelBytes, modelIsCached, removeCachedModel, verifyCachedModel } from '@/lib/model-cache';
import { backendCandidatesForRuntime } from '@/lib/backend-order';
import { formatBytes, MODEL_CATALOG, type ModelSpec } from '@/lib/models';

type CacheMap = Record<string, boolean>;

function supportedBackends(): string[] {
  if (typeof window === 'undefined') return [];
  const canvas = document.createElement('canvas');
  const hasWebGl = Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'));
  return backendCandidatesForRuntime(navigator, hasWebGl).map((backend) => backend.toUpperCase());
}

export function ModelsPanel({ runtime }: { runtime: EdgeRuntime }) {
  const [cache, setCache] = useState<CacheMap>({});
  const [activeId, setActiveId] = useState('');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');

  const refresh = useCallback(async () => {
    const entries = await Promise.all(MODEL_CATALOG.map(async (model) => [model.id, await modelIsCached(model)] as const));
    setCache(Object.fromEntries(entries));
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  async function download(model: ModelSpec) {
    setActiveId(model.id);
    setProgress(0);
    setMessage(`Preparing ${model.title}`);
    try {
      const artifact = await ensureModelBytes(model, setProgress);
      const valid = await verifyCachedModel(model);
      if (!valid) throw new Error('Cached model failed integrity verification.');
      setMessage(`${model.title} verified from ${artifact.source}.`);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Model download failed.');
    } finally {
      setActiveId('');
    }
  }

  async function loadTest(model: ModelSpec) {
    setActiveId(model.id);
    setMessage(`Compiling ${model.title}`);
    runtime.selectModel(model.id);
    try {
      const result = await runtime.loadModel(model.id);
      setMessage(`${model.title} loaded with ${String(result.backend).toUpperCase()}.`);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Model load test failed.');
    } finally {
      setActiveId('');
    }
  }

  async function remove(model: ModelSpec) {
    setActiveId(model.id);
    runtime.engine.release();
    await removeCachedModel(model);
    setMessage(`${model.title} removed from this device.`);
    await refresh();
    setActiveId('');
  }

  const backends = supportedBackends();

  return (
    <section className="grid flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
      <Card className="min-h-[65svh] border-white/10 bg-card/60 ring-0">
        <CardHeader className="border-b border-white/8 pb-4">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-300">Model library</p>
          <CardTitle className="mt-1 text-xl">Verified local artifacts</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 p-4 sm:grid-cols-2 sm:p-6">
          {MODEL_CATALOG.map((model) => (
            <article key={model.id} className={`relative overflow-hidden rounded-[22px] border p-4 sm:p-5 ${cache[model.id] ? 'border-emerald-300/18 bg-emerald-300/[0.035]' : 'border-white/8 bg-white/[0.025]'}`}>
              <div className="mb-5 flex items-start justify-between gap-3">
                <div className="grid size-12 place-items-center rounded-2xl bg-cyan-300/8"><Box className="size-6 text-cyan-300" /></div>
                <Badge variant="outline" className={cache[model.id] ? 'border-emerald-300/20 bg-emerald-300/8 text-emerald-300' : 'border-white/10 text-muted-foreground'}>
                  {cache[model.id] ? <CheckCircle2 /> : <DownloadCloud />}{cache[model.id] ? 'Cached' : 'Remote'}
                </Badge>
              </div>
              <h3 className="text-lg font-medium">{model.title}</h3>
              <p className="mt-1 text-sm text-muted-foreground">LoRA merged · static ONNX · opset 12</p>
              <dl className="mt-5 grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-xl bg-black/10 p-3"><dt className="text-muted-foreground">Artifact</dt><dd className="mt-1 font-mono">{formatBytes(model.sizeBytes)}</dd></div>
                <div className="rounded-xl bg-black/10 p-3"><dt className="text-muted-foreground">Input</dt><dd className="mt-1 font-mono">640×640</dd></div>
                <div className="col-span-2 rounded-xl bg-black/10 p-3"><dt className="text-muted-foreground">SHA-256</dt><dd className="mt-1 truncate font-mono" title={model.sha256}>{model.sha256.slice(0, 20)}…</dd></div>
              </dl>
              {activeId === model.id && (
                <div className="mt-4"><div className="mb-1 flex justify-between text-[11px] text-muted-foreground"><span>Working</span><span>{progress}%</span></div><div className="h-1.5 overflow-hidden rounded-full bg-white/8"><div className="h-full bg-emerald-300 transition-all" style={{ width: `${progress}%` }} /></div></div>
              )}
              <div className="mt-5 flex flex-wrap gap-2">
                <Button type="button" size="sm" onClick={() => void download(model)} disabled={Boolean(activeId)} className="bg-cyan-300 text-cyan-950 hover:bg-cyan-200">{activeId === model.id ? <LoaderCircle className="animate-spin" /> : <DownloadCloud />}{cache[model.id] ? 'Verify' : 'Download'}</Button>
                <Button type="button" size="sm" variant="outline" onClick={() => void loadTest(model)} disabled={Boolean(activeId)} className="border-white/10"><Cpu /> Load test</Button>
                <Button type="button" size="icon-sm" variant="destructive" onClick={() => void remove(model)} disabled={!cache[model.id] || Boolean(activeId)}><Trash2 /><span className="sr-only">Delete cached model</span></Button>
              </div>
            </article>
          ))}
        </CardContent>
      </Card>

      <aside className="flex flex-col gap-3">
        <Card className="border-white/10 bg-card/60 ring-0"><CardContent className="p-4"><div className="mb-4 flex items-center gap-2 font-medium"><Cpu className="size-4 text-violet-300" /> Browser backends</div><div className="flex flex-wrap gap-2">{backends.map((backend, index) => <Badge key={backend} variant="outline" className={index === 0 ? 'border-cyan-300/20 bg-cyan-300/8 text-cyan-300' : 'border-white/10 text-muted-foreground'}>{backend}</Badge>)}</div><p className="mt-4 text-xs leading-5 text-muted-foreground">The app tries the fastest compatible provider and falls back automatically. iPhone uses WASM first for reliable model support.</p></CardContent></Card>
        <Card className="border-white/10 bg-card/60 ring-0"><CardContent className="p-4"><div className="mb-4 flex items-center gap-2 font-medium"><Database className="size-4 text-emerald-300" /> Device storage</div><p className="text-sm leading-6 text-muted-foreground">Model bytes are stored in IndexedDB after SHA-256 verification. Photos are never persisted by the app.</p><div className="mt-4 flex items-center gap-2 text-xs text-emerald-300"><ShieldCheck className="size-4" /> Privacy-first local inference</div></CardContent></Card>
        <Card className="border-white/10 bg-card/60 ring-0"><CardContent className="p-4"><div className="mb-3 flex items-center gap-2 font-medium"><HardDrive className="size-4 text-cyan-300" /> Cache state</div><p className="font-mono text-3xl">{Object.values(cache).filter(Boolean).length}<span className="ml-2 text-sm text-muted-foreground">/ {MODEL_CATALOG.length}</span></p><Button type="button" variant="ghost" size="sm" onClick={() => void refresh()} className="mt-3"><RefreshCw /> Refresh</Button></CardContent></Card>
        {message && <div className="rounded-xl border border-white/8 bg-white/[0.035] p-3 text-sm text-slate-300">{message}</div>}
      </aside>
    </section>
  );
}
