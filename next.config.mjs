/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    // Only proxy to localhost:8000 during local development
    if (process.env.NODE_ENV === 'development') {
      return [
        {
          source: '/api/:path*',
          destination: 'http://127.0.0.1:8000/api/:path*',
        },
      ];
    }
    // In production (Vercel), use the serverless functions in /api directly
    return [];
  },
};

export default nextConfig;
