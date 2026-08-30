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
  title: 'C3 Edge Lab',
  description: 'Installable on-device industrial defect inference and benchmark lab.',
  applicationName: 'C3 Edge Lab',
  manifest: '/manifest.webmanifest',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'C3 Edge',
  },
  formatDetection: {
    telephone: false,
  },
  icons: {
    icon: '/icon.svg',
    apple: '/icon-192.png',
  },
  openGraph: {
    type: 'website',
    url: '/',
    title: 'C3 Edge Lab',
    description: 'Private on-device industrial defect inference and benchmark lab.',
    siteName: 'C3 Edge Lab',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'C3 Edge Lab on-device defect inspection' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'C3 Edge Lab',
    description: 'Private on-device industrial defect inference and benchmark lab.',
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
