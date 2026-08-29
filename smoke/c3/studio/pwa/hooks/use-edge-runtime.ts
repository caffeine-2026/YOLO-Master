'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import { EdgeInferenceEngine } from '@/lib/inference-engine';
import { MODEL_CATALOG, modelById } from '@/lib/models';

export type RuntimePhase = 'idle' | 'downloading' | 'compiling' | 'ready' | 'error';

export function useEdgeRuntime() {
  const engineRef = useRef<EdgeInferenceEngine | null>(null);
  if (!engineRef.current) engineRef.current = new EdgeInferenceEngine();
  const [selectedId, setSelectedIdState] = useState(MODEL_CATALOG[0].id);
  const [phase, setPhase] = useState<RuntimePhase>('idle');
  const [progress, setProgress] = useState(0);
  const [backend, setBackend] = useState<string>('Not loaded');
  const [error, setError] = useState('');

  useEffect(() => () => engineRef.current?.release(), []);

  const selectModel = useCallback((id: string) => {
    engineRef.current?.release();
    setSelectedIdState(id);
    setPhase('idle');
    setProgress(0);
    setBackend('Not loaded');
    setError('');
  }, []);

  const loadModel = useCallback(async (modelId = selectedId) => {
    const model = modelById(modelId);
    setError('');
    setPhase('downloading');
    setProgress(0);
    try {
      const result = await engineRef.current!.load(model, (value) => {
        setProgress(value);
        if (value >= 100) setPhase('compiling');
      });
      setProgress(100);
      setBackend(String(result.backend ?? 'unknown').toUpperCase());
      setPhase('ready');
      return result;
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : 'Unable to load the model.';
      setError(message);
      setPhase('error');
      throw loadError;
    }
  }, [selectedId]);

  return {
    engine: engineRef.current,
    model: modelById(selectedId),
    selectedId,
    selectModel,
    loadModel,
    phase,
    progress,
    backend,
    error,
    setError,
  };
}

export type EdgeRuntime = ReturnType<typeof useEdgeRuntime>;
