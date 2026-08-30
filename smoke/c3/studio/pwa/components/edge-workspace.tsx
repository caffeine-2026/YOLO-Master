'use client';

import Image from 'next/image';
import { useEffect, useState } from 'react';
import { Camera, ChartNoAxesColumnIncreasing, Download, Images, Layers3, Share } from 'lucide-react';

import { BenchPanel } from '@/components/bench-panel';
import { LivePanel } from '@/components/live-panel';
import { ModelsPanel } from '@/components/models-panel';
import { PhotoPanel } from '@/components/photo-panel';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useEdgeRuntime } from '@/hooks/use-edge-runtime';
import { useInstallPrompt } from '@/hooks/use-install-prompt';

type ViewId = 'live' | 'photo' | 'bench' | 'models';

const navigation = [
  { id: 'live' as const, label: 'Live', icon: Camera },
  { id: 'photo' as const, label: 'Photo', icon: Images },
  { id: 'bench' as const, label: 'Bench', icon: ChartNoAxesColumnIncreasing },
  { id: 'models' as const, label: 'Models', icon: Layers3 },
];

export function EdgeWorkspace() {
  const runtime = useEdgeRuntime();
  const installer = useInstallPrompt();
  const [activeView, setActiveView] = useState<ViewId>('live');
  const [online, setOnline] = useState(() => typeof navigator === 'undefined' || navigator.onLine);

  useEffect(() => {
    const update = () => setOnline(navigator.onLine);
    window.addEventListener('online', update);
    window.addEventListener('offline', update);
    return () => {
      window.removeEventListener('online', update);
      window.removeEventListener('offline', update);
    };
  }, []);

  return (
    <main className="min-h-svh bg-background text-foreground">
      <div className="mx-auto flex min-h-svh w-full max-w-[1480px] flex-col px-3 pb-[calc(86px+env(safe-area-inset-bottom))] pt-3 sm:px-6 sm:pt-5 lg:pb-6">
        <header className="mb-3 flex items-center justify-between gap-3 sm:mb-5">
          <div className="flex min-w-0 items-center gap-3">
            <div className="relative size-10 shrink-0 overflow-hidden rounded-[14px] border border-cyan-300/20 bg-cyan-300/10 shadow-[0_0_30px_rgb(34_211_238/10%)]">
              <Image src="/icon-192.png" alt="YOLO-V-PEFT penguin inspector" width={40} height={40} className="size-full object-cover" priority />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h1 className="truncate text-base font-semibold tracking-[-0.02em] sm:text-lg">YOLO-V-PEFT</h1>
                <Badge className="border border-emerald-300/20 bg-emerald-300/10 text-emerald-300">PWA</Badge>
                <span className={`hidden size-2 rounded-full sm:block ${online ? 'bg-emerald-400' : 'bg-amber-400'}`} title={online ? 'Online' : 'Offline'} />
              </div>
              <p className="truncate text-xs text-muted-foreground">C3 V-PEFT research benchmark · on-device</p>
            </div>
          </div>
          {!installer.isStandalone && (
            <Button type="button" onClick={() => void installer.install()} variant="outline" className="h-9 border-white/10 bg-white/[0.035] px-3 text-xs">
              <Download />
              <span className="hidden sm:inline">Install app</span>
            </Button>
          )}
        </header>

        {activeView === 'live' && <LivePanel runtime={runtime} />}
        {activeView === 'photo' && <PhotoPanel runtime={runtime} />}
        {activeView === 'bench' && <BenchPanel runtime={runtime} />}
        {activeView === 'models' && <ModelsPanel runtime={runtime} />}

        <nav className="fixed inset-x-3 bottom-[calc(10px+env(safe-area-inset-bottom))] z-30 mx-auto grid max-w-md grid-cols-4 rounded-[22px] border border-white/10 bg-[#0b151f]/90 p-1.5 shadow-2xl backdrop-blur-xl lg:static lg:mt-4 lg:max-w-xl">
          {navigation.map((item) => {
            const Icon = item.icon;
            const active = activeView === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveView(item.id)}
                aria-current={active ? 'page' : undefined}
                className={`flex min-h-14 flex-col items-center justify-center gap-1 rounded-2xl text-[11px] font-medium transition ${active ? 'bg-cyan-300/12 text-cyan-300' : 'text-slate-500 hover:text-slate-200'}`}
              >
                <Icon className="size-[18px]" />
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>

      <Dialog open={installer.showIosHelp} onOpenChange={(open) => { if (!open) installer.closeIosHelp(); }}>
        <DialogContent className="border-white/10 bg-[#0b151f] text-slate-100">
          <DialogHeader>
            <div className="mb-2 grid size-11 place-items-center rounded-xl bg-cyan-300/10"><Share className="size-5 text-cyan-300" /></div>
            <DialogTitle>Install YOLO-V-PEFT</DialogTitle>
            <DialogDescription className="leading-6 text-slate-400">On iPhone, open this page in Safari, tap the Share button, then choose <strong className="text-slate-200">Add to Home Screen</strong>. On desktop, use the install icon in the address bar.</DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </main>
  );
}
