import {defineConfig} from 'vite';

const target = 'http://127.0.0.1:8000';
const routes = ['/health', '/claims', '/detect', '/retrieve', '/verify', '/metrics', '/charts'];

export default defineConfig({
  server: {
    host: '127.0.0.1',
    proxy: Object.fromEntries(routes.map(route => [route, target])),
  },
});
