import type { Metadata, Viewport } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  metadataBase: new URL('https://yoloc3vpeft.com'),
  title: 'YOLO-V-PEFT',
  description: 'On-device runtime and dataset-scope inference for the C3 V-PEFT research benchmark.',
  applicationName: 'YOLO-V-PEFT',
  manifest: '/manifest.webmanifest',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'YOLO-V-PEFT',
  },
  formatDetection: {
    telephone: false,
  },
  icons: {
    icon: [
      { url: '/icon-64.png', sizes: '64x64', type: 'image/png' },
      { url: '/icon-192.png', sizes: '192x192', type: 'image/png' },
    ],
    apple: [{ url: '/icon-180.png', sizes: '180x180', type: 'image/png' }],
  },
  openGraph: {
    type: 'website',
    url: '/',
    title: 'YOLO-V-PEFT',
    description: 'On-device runtime and dataset-scope inference for the C3 V-PEFT research benchmark.',
    siteName: 'YOLO-V-PEFT',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'YOLO-V-PEFT research benchmark' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'YOLO-V-PEFT',
    description: 'On-device runtime and dataset-scope inference for the C3 V-PEFT research benchmark.',
    images: ['/og.png'],
  },
};

export const viewport: Viewport = {
  themeColor: '#071019',
  colorScheme: 'dark',
  viewportFit: 'cover',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko" className="dark">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>{children}</body>
    </html>
  );
}
