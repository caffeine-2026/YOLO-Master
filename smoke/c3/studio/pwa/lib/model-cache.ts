import type { ModelSpec } from '@/lib/models';

const DATABASE_NAME = 'c3-edge-models-v1';
const STORE_NAME = 'models';

type CachedModel = {
  key: string;
  modelId: string;
  sha256: string;
  bytes: ArrayBuffer;
  cachedAt: string;
};

function cacheKey(model: ModelSpec): string {
  return `${model.id}:${model.sha256}`;
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) database.createObjectStore(STORE_NAME, { keyPath: 'key' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Unable to open the model cache.'));
  });
}

function transactionRequest<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDatabase().then(
    (database) =>
      new Promise<T>((resolve, reject) => {
        const transaction = database.transaction(STORE_NAME, mode);
        const request = action(transaction.objectStore(STORE_NAME));
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error('Model cache operation failed.'));
        transaction.oncomplete = () => database.close();
      }),
  );
}

async function sha256(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, '0')).join('');
}

async function downloadModel(model: ModelSpec, onProgress: (progress: number) => void): Promise<ArrayBuffer> {
  const response = await fetch(model.url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Model download failed with HTTP ${response.status}.`);
  const expectedBytes = Number(response.headers.get('content-length')) || model.sizeBytes;
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    onProgress(100);
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress(Math.min(99, Math.round((received / expectedBytes) * 100)));
  }
  const combined = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  onProgress(100);
  return combined.buffer;
}

export async function modelIsCached(model: ModelSpec): Promise<boolean> {
  const record = await transactionRequest<CachedModel | undefined>('readonly', (store) => store.get(cacheKey(model)));
  return Boolean(record?.bytes?.byteLength);
}

export async function ensureModelBytes(
  model: ModelSpec,
  onProgress: (progress: number) => void = () => undefined,
): Promise<{ bytes: ArrayBuffer; source: 'cache' | 'network' }> {
  const cached = await transactionRequest<CachedModel | undefined>('readonly', (store) => store.get(cacheKey(model)));
  if (cached?.bytes?.byteLength) {
    onProgress(100);
    return { bytes: cached.bytes, source: 'cache' };
  }

  const bytes = await downloadModel(model, onProgress);
  const digest = await sha256(bytes);
  if (digest.toLowerCase() !== model.sha256.toLowerCase()) {
    throw new Error('Downloaded model failed SHA-256 verification.');
  }
  const record: CachedModel = {
    key: cacheKey(model),
    modelId: model.id,
    sha256: digest,
    bytes,
    cachedAt: new Date().toISOString(),
  };
  await transactionRequest<IDBValidKey>('readwrite', (store) => store.put(record));
  return { bytes, source: 'network' };
}

export async function verifyCachedModel(model: ModelSpec): Promise<boolean> {
  const cached = await transactionRequest<CachedModel | undefined>('readonly', (store) => store.get(cacheKey(model)));
  if (!cached?.bytes?.byteLength) return false;
  return (await sha256(cached.bytes)).toLowerCase() === model.sha256.toLowerCase();
}

export async function removeCachedModel(model: ModelSpec): Promise<void> {
  await transactionRequest<undefined>('readwrite', (store) => store.delete(cacheKey(model)));
}
