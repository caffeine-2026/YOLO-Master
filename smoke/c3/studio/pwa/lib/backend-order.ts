export type EdgeBackend = 'webgl' | 'wasm';

export type RuntimeNavigatorInfo = {
  userAgent?: string;
  platform?: string;
  maxTouchPoints?: number;
};

export function isAppleMobileRuntime(info: RuntimeNavigatorInfo): boolean {
  const userAgent = info.userAgent ?? '';
  const platform = info.platform ?? '';
  const maxTouchPoints = info.maxTouchPoints ?? 0;
  return /iPhone|iPad|iPod/i.test(userAgent)
    || (/Mac/i.test(platform) && maxTouchPoints > 1);
}

export function backendCandidatesForRuntime(
  info: RuntimeNavigatorInfo,
  hasWebGl: boolean,
): EdgeBackend[] {
  if (!hasWebGl) return ['wasm'];
  return isAppleMobileRuntime(info) ? ['wasm', 'webgl'] : ['webgl', 'wasm'];
}
