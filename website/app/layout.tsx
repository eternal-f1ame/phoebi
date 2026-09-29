import type { Metadata, Viewport } from "next";
import { Fraunces, Inter, JetBrains_Mono } from "next/font/google";
import type { ReactNode } from "react";
import { site } from "@/content";
import "./globals.css";

const serif = Fraunces({
  subsets: ["latin"],
  variable: "--font-serif",
  display: "swap",
  axes: ["opsz"],
  style: ["normal", "italic"],
});
const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

const preview = { url: "/img/og.png", width: 1200, height: 630, alt: `${site.short}: ${site.venue}` };

export const metadata: Metadata = {
  metadataBase: new URL(site.url),
  title: site.title,
  description: site.description,
  openGraph: {
    type: "website",
    url: site.url,
    siteName: site.short,
    title: site.title,
    description: site.description,
    images: [preview],
  },
  twitter: { card: "summary_large_image", title: site.title, description: site.description, images: [preview] },
};

export const viewport: Viewport = { themeColor: "#ebe7de" };

// Marks the page as scripted before its first paint, so an animated diagram can start empty
// without readers who have no JavaScript seeing it empty. The class is added outside React,
// hence suppressHydrationWarning on <html>.
const markScripted = "document.documentElement.classList.add('js')";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: markScripted }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
