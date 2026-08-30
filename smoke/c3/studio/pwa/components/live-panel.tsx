'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  Aperture,
  Cpu,
  FlipHorizontal2,
  Pause,
  Play,
  Settings2,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { NativeSelect, NativeSelectOption } from '@/components/ui/native-select';
import type { EdgeRuntime } from '@/hooks/use-edge-runtime';
import { drawDetections } from '@/lib/image';
import type { InferenceTimings } from '@/lib/inference-engine';
import { MODEL_CATALOG } from '@/lib/models';

const EMPTY_TIMINGS: InferenceTimings = {
  preprocessMs: 0,
  inferenceMs: 0,
  postprocessMs: 0,
  totalMs: 0,
};

export function LivePanel({ runtime }: { runtime: EdgeRuntime }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastCompletedRef = useRef(0);
  const [running, setRunning] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [facingMode, setFacingMode] = useState<'environment' | 'user'>('environment');
  const [iou] = useState(45);
  const [fps, setFps] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);
  const [timings, setTimings] = useState<InferenceTimings>(EMPTY_TIMINGS);
  const [cameraError, setCameraError] = useState('');

  const stopStream = useCallback(() => {
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    if (overlayRef.current) drawDetections(overlayRef.current, [], 1, 1);
    setCameraReady(false);
    setRunning(false);
    setDetectionCount(0);
    setFps(0);
  }, []);

  const openCamera = useCallback(async (position: 'environment' | 'user') => {
    setCameraError('');
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: position },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = stream;
    if (!videoRef.current) throw new Error('Camera preview is unavailable.');
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
    setCameraReady(true);
  }, []);

  useEffect(() => stopStream, [stopStream]);

  useEffect(() => {
    if (!running || runtime.phase !== 'ready') return;
    let cancelled = false;
    const processFrame = async () => {
      if (cancelled || !videoRef.current || !overlayRef.current) return;
      const video = videoRef.current;
      if (video.readyState >= 2 && video.videoWidth > 0) {
        try {
          const result = await runtime.engine.runSource(
            video,
            video.videoWidth,
            video.videoHeight,
            runtime.model.recommendedConfidence,
            iou / 100,
          );
          if (cancelled) return;
          drawDetections(overlayRef.current, result.detections, result.width, result.height, 'cover');
          const completed = performance.now();
          const observedFps = lastCompletedRef.current
            ? 1000 / Math.max(1, completed - lastCompletedRef.current)
            : 1000 / Math.max(1, result.timings.totalMs);
          lastCompletedRef.current = completed;
          setFps(observedFps);
          setDetectionCount(result.detections.length);
          setTimings(result.timings);
        } catch (error) {
          setCameraError(error instanceof Error ? error.message : 'Live inference failed.');
          setRunning(false);
          return;
        }
      }
      animationRef.current = requestAnimationFrame(processFrame);
    };
    animationRef.current = requestAnimationFrame(processFrame);
    return () => {
      cancelled = true;
      if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    };
  }, [iou, running, runtime.engine, runtime.model.recommendedConfidence, runtime.phase]);

  async function toggleRun() {
    if (running) {
      stopStream();
      return;
    }
    try {
      await openCamera(facingMode);
      await runtime.loadModel();
      lastCompletedRef.current = 0;
      setRunning(true);
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : 'Unable to start the camera.');
      stopStream();
    }
  }

  async function switchCamera() {
    const next = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(next);
    if (!cameraReady) return;
    try {
      await openCamera(next);
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : 'Unable to switch cameras.');
    }
  }

  async function changeModel(id: string) {
    const resume = running;
    setRunning(false);
    runtime.selectModel(id);
    if (resume) {
      try {
        await runtime.loadModel(id);
        lastCompletedRef.current = 0;
        setRunning(true);
      } catch {
        stopStream();
      }
    }
  }

  const statusText = runtime.phase === 'downloading'
    ? `Downloading ${runtime.progress}%`
    : runtime.phase === 'compiling'
      ? 'Compiling model'
      : runtime.phase === 'ready'
        ? `${runtime.backend} · on-device`
        : 'Model not loaded';

  return (
    <section className="grid flex-1 gap-3 lg:grid-cols-[minmax(0,1fr)_330px] lg:gap-4">
      <Card className="relative min-h-[65svh] overflow-hidden rounded-[26px] border border-white/10 bg-[#050b11] p-0 ring-0 sm:min-h-[70svh] lg:min-h-0">
        <CardContent className="relative h-full min-h-[65svh] p-0 sm:min-h-[70svh] lg:min-h-0">
          <video
            ref={videoRef}
            muted
            playsInline
            className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-500 ${runtime.model.inputColorMode === 'grayscale' ? 'grayscale' : ''} ${cameraReady ? 'opacity-100' : 'opacity-0'}`}
          />
          <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-label="Detection overlay" />
          {!cameraReady && <div className="camera-grid absolute inset-0" aria-hidden="true" />}
          <div className="absolute inset-x-0 top-0 z-10 flex items-start justify-between gap-3 p-3 sm:p-5">
            <div className="glass-panel flex min-w-0 items-center gap-2 rounded-2xl p-2">
              <NativeSelect
                aria-label="Detection model"
                value={runtime.selectedId}
                onChange={(event) => void changeModel(event.target.value)}
                className="min-w-0"
              >
                {MODEL_CATALOG.map((model) => (
                  <NativeSelectOption key={model.id} value={model.id}>{model.title}</NativeSelectOption>
                ))}
              </NativeSelect>
              <Badge variant="outline" className="hidden border-white/10 bg-black/20 text-slate-300 sm:flex">
                <Cpu /> {statusText}
              </Badge>
            </div>
            <Button
              type="button"
              onClick={() => void switchCamera()}
              variant="outline"
              size="icon-lg"
              className="glass-panel rounded-2xl border-white/10 text-slate-200"
            >
              <FlipHorizontal2 />
              <span className="sr-only">Switch camera</span>
            </Button>
          </div>

          {!cameraReady && (
            <div className="absolute inset-0 grid place-items-center px-6 text-center">
              <div className="max-w-sm">
                <div className="mx-auto mb-5 grid size-20 place-items-center rounded-full border border-cyan-300/20 bg-cyan-300/[0.07] shadow-[0_0_70px_rgb(34_211_238/12%)]">
                  <Aperture className="size-9 text-cyan-300" />
                </div>
                <p className="text-lg font-medium tracking-tight">Camera is ready when you are</p>
                <p className="mt-2 text-sm leading-6 text-slate-400">Frames stay on this device. The model is downloaded once, verified, and cached locally.</p>
              </div>
            </div>
          )}

          {(cameraError || runtime.error) && (
            <div className="absolute inset-x-3 top-24 z-20 rounded-xl border border-rose-400/20 bg-rose-950/80 p-3 text-sm text-rose-200 backdrop-blur sm:inset-x-5">
              {cameraError || runtime.error}
            </div>
          )}

          {(runtime.phase === 'downloading' || runtime.phase === 'compiling') && (
            <div className="absolute inset-x-3 top-24 z-20 rounded-xl border border-cyan-300/15 bg-[#071019]/90 p-3 backdrop-blur sm:inset-x-5">
              <div className="mb-2 flex justify-between text-xs"><span>{statusText}</span><span>{runtime.progress}%</span></div>
              <div className="h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full bg-cyan-300 transition-all" style={{ width: `${runtime.progress}%` }} /></div>
            </div>
          )}

          <div className="absolute inset-x-0 bottom-0 z-10 p-3 sm:p-5">
            <div className="glass-panel grid grid-cols-[auto_1fr_auto] items-center gap-3 rounded-[22px] p-3 sm:gap-5 sm:p-4">
              <div className="grid size-16 place-items-center rounded-full border border-violet-300/20 bg-violet-300/10 sm:size-20">
                <div className="text-center">
                  <strong className="block font-mono text-2xl text-violet-200 sm:text-3xl">{fps.toFixed(1)}</strong>
                  <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-violet-300/70">FPS</span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs sm:grid-cols-4">
                {[
                  ['Pre', timings.preprocessMs],
                  ['Model', timings.inferenceMs],
                  ['NMS', timings.postprocessMs],
                  ['Total', timings.totalMs],
                ].map(([label, value]) => (
                  <div key={String(label)}>
                    <span className="block text-slate-500">{label}</span>
                    <span className="font-mono text-slate-200">{Number(value).toFixed(1)} ms</span>
                  </div>
                ))}
              </div>
              <Button
                type="button"
                onClick={() => void toggleRun()}
                disabled={runtime.phase === 'downloading' || runtime.phase === 'compiling'}
                className={`size-14 rounded-full sm:size-16 ${running ? 'bg-rose-400 text-rose-950 hover:bg-rose-300' : 'bg-cyan-300 text-cyan-950 hover:bg-cyan-200'}`}
              >
                {running ? <Pause className="size-6" /> : <Play className="size-6 fill-current" />}
                <span className="sr-only">{running ? 'Stop inference' : 'Start inference'}</span>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <aside className="hidden min-h-0 flex-col gap-3 lg:flex">
        <Card className="border-white/10 bg-card/70 ring-0">
          <CardContent className="p-4">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="text-xs font-medium uppercase tracking-[0.14em] text-cyan-300">Runtime</p>
                <h2 className="mt-1 text-lg font-medium">Edge session</h2>
              </div>
              <span className={`relative flex size-3 rounded-full ${running ? 'bg-emerald-400' : 'bg-slate-600'}`} />
            </div>
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between gap-3"><dt className="text-muted-foreground">Model</dt><dd>{runtime.model.shortTitle}</dd></div>
              <div className="flex justify-between gap-3"><dt className="text-muted-foreground">Backend</dt><dd>{runtime.backend}</dd></div>
              <div className="flex justify-between gap-3"><dt className="text-muted-foreground">Input</dt><dd className="font-mono">1×3×640×640</dd></div>
              <div className="flex justify-between gap-3"><dt className="text-muted-foreground">Review candidates</dt><dd className="font-mono text-amber-300">{detectionCount}</dd></div>
            </dl>
          </CardContent>
        </Card>
        <Card className="flex-1 border-white/10 bg-card/70 ring-0">
          <CardContent className="flex h-full flex-col p-4">
            <div className="flex items-center gap-2 text-sm font-medium"><Activity className="size-4 text-violet-300" /> Session trace</div>
            <div className="my-auto py-8 text-center">
              <div className="mx-auto mb-3 grid size-11 place-items-center rounded-xl bg-white/[0.04]"><Settings2 className="size-5 text-slate-500" /></div>
              <p className="text-sm text-slate-400">{running ? `${detectionCount} review candidates · ${timings.totalMs.toFixed(1)} ms/frame` : runtime.model.captureHint}</p>
            </div>
          </CardContent>
        </Card>
      </aside>
    </section>
  );
}
