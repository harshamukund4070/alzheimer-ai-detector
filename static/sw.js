// NeuroScan AI Service Worker
const CACHE_NAME = 'neuroscan-v1';
const STATIC_ASSETS = [
  '/',
  '/upload/',
  '/manifest.json',
  '/sw.js',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  '/static/icons/screenshot1.png'
];

// Install: cache core assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(STATIC_ASSETS).catch(err => {
        console.warn('Cache addAll failed:', err);
      });
    })
  );
  self.skipWaiting();
});

// Activate: clear old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// Fetch: network-first strategy, fallback to cache
self.addEventListener('fetch', event => {
  // Skip non-GET or cross-origin
  if (event.request.method !== 'GET') return;
  if (!event.request.url.startsWith(self.location.origin)) return;

  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Clone and cache fresh response
        const clone = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
