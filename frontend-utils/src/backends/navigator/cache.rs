//! AQW build: the navigator's side of the disk cache (`ruffle_http_cache`): keeps a response's
//! body as it is read.

use std::sync::Arc;
use url::Url;

pub use ruffle_http_cache::{CacheStats, Entry, HttpCache, MAX_BODY, Policy};

/// Collects a response's body as it is read, and stores it when all of it has arrived.
pub(crate) struct CacheWriter {
    cache: Arc<HttpCache>,
    url: Url,
    policy: Policy,
    body: Vec<u8>,
}

impl CacheWriter {
    pub(crate) fn new(cache: Arc<HttpCache>, url: Url, policy: Policy) -> Self {
        let capacity = policy.len.unwrap_or(0).min(MAX_BODY) as usize;
        Self {
            cache,
            url,
            policy,
            body: Vec::with_capacity(capacity),
        }
    }

    /// Adds a chunk. `false`: the body is larger than the cache takes, drop the writer.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> bool {
        if (self.body.len() + chunk.len()) as u64 > MAX_BODY {
            return false;
        }
        self.body.extend_from_slice(chunk);
        true
    }

    /// The body is complete: writes the entry, off this thread where there's a Tokio runtime.
    pub(crate) fn finish(self) {
        // A connection that closed early gives a short body without an error.
        if self
            .policy
            .len
            .is_some_and(|len| len != self.body.len() as u64)
        {
            return;
        }
        let store = move || self.cache.store(&self.url, &self.policy, &self.body);
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                runtime.spawn_blocking(store);
            }
            Err(_) => store(),
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use reqwest::header::HeaderMap;

    #[test]
    fn writer_stores_only_whole_bodies() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HttpCache::new(dir.path(), 1 << 30).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert("cache-control", "public, max-age=2678400".parse().unwrap());
        headers.insert("etag", "\"v1\"".parse().unwrap());
        headers.insert("content-length", "5".parse().unwrap());
        let policy = Policy::of_response(&headers).unwrap();
        let url = Url::parse("https://game.aq.com/game/gamefiles/w.swf").unwrap();

        let mut short = CacheWriter::new(cache.clone(), url.clone(), policy.clone());
        assert!(short.push(b"hel"));
        short.finish();
        assert!(cache.lookup(&url).is_none());

        let mut whole = CacheWriter::new(cache.clone(), url.clone(), policy);
        assert!(whole.push(b"hel"));
        assert!(whole.push(b"lo"));
        whole.finish();
        assert_eq!(cache.lookup(&url).unwrap().body, b"hello");
        assert_eq!(cache.stats().stored, 1);
    }
}
