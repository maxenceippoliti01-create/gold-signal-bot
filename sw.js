// Gold Trader Pro — Service Worker
const CACHE_NAME = 'gtp-v1';
const CORE_FILES = [
  './gold_trading.html',
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
];

// Installation : mise en cache des fichiers core
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(CORE_FILES).catch(() => {});
    })
  );
  self.skipWaiting();
});

// Activation : supprime les anciens caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// Fetch : cache-first pour les fichiers locaux, network-first pour les APIs
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // APIs externes → toujours réseau (pas de cache)
  if (
    url.hostname.includes('finnhub.io') ||
    url.hostname.includes('tradingview.com') ||
    url.hostname.includes('faireconomy.media')
  ) {
    event.respondWith(fetch(event.request).catch(() => new Response('', { status: 503 })));
    return;
  }

  // Fichiers locaux → cache-first avec fallback réseau
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => caches.match('./gold_trading.html'));
    })
  );
});
