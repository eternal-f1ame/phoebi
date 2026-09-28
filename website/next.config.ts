import type { NextConfig } from "next";

// A fully static site: `next build` writes it to out/, which Vercel serves as is.
const config: NextConfig = {
  output: "export",
  images: { unoptimized: true },
};

export default config;
