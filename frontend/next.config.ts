import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow images served from the Python backend
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000",
      },
    ],
  },
};

export default nextConfig;
