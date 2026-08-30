import { readFile, writeFile } from 'node:fs/promises';

const configUrl = new URL('../dist/server/wrangler.json', import.meta.url);
const config = JSON.parse(await readFile(configUrl, 'utf8'));

config.workers_dev = false;
config.preview_urls = false;

await writeFile(configUrl, `${JSON.stringify(config)}\n`, 'utf8');
