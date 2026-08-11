import type { NextConfig } from "next";

const basePath = process.env.DYNNAV_SITE_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  basePath,
  assetPrefix: basePath || undefined,
  images: { unoptimized: true },
  turbopack: { root: process.cwd() },
};

export default nextConfig;
