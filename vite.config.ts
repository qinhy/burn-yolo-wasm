import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [vue()],
  base: './',
  worker: { format: 'es' },
  build: { target: 'esnext', assetsInlineLimit: 0, rollupOptions: { input: { main: 'index.html', video: 'video.html' } } },
  server: { port: 5173, strictPort: true },
});
