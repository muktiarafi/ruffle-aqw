//! AQW build: a disk cache for the files the game downloads, used by ruffle-aqw's navigator
//! (`ruffle_frontend_utils`) and by frontends that fetch files themselves. No Ruffle player here,
//! so a frontend without one (Turu Mas, on Adobe AIR) can share the cache.
//!
//! In a browser, Flash Player's downloads went through the browser's cache; Ruffle's desktop
//! navigator has none, so every start fetched the game's assets again (8 MB) and every map join
//! its map, monsters and gear. AQW's CDN marks those files `Cache-Control: public,
//! max-age=2678400` (31 days) with an `ETag`, so this follows the same rules a browser does:
//!
//! * Only a `GET` without a body or headers of its own, answered `200` with an explicit
//!   `max-age` of an hour or more, a known length, no `Vary` beyond `Accept-Encoding`, and no
//!   `Set-Cookie` unless it is also `public`, is kept (the cookie isn't: an entry keeps no
//!   headers). AQW's CDN sets Cloudflare's `__cflb` on any request that comes without it, a
//!   client's first at least, on files that are `public`. The game's API answers (`max-age=2`,
//!   `Vary: *`) never are.
//! * A file is one entry whatever its query string, and remembers the query it came with.
//!   Asked for with the same query while fresh, it is answered from disk with no request.
//!   With another query (the game's `?v=5.32` after an update, skua.swf's `?ver=<random>`),
//!   or once stale, it is revalidated: a conditional request the server answers `304` without
//!   a body when nothing changed.
//! * When the network fails and an entry exists, the entry is used.
//!
//! An entry is one file, `<fnv64 of the URL>.entry`: a [`HEADER_LEN`]-byte text header (padded
//! with spaces, so a revalidation rewrites it in place) and then the body. Entries are written
//! to a temporary file and renamed, so other clients sharing the folder never see half of one.
//! The folder is kept under `max_bytes` by removing the entries used longest ago.

use http::header::{self, HeaderMap};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use url::Url;

/// The size of an entry's header. A header that doesn't fit (a very long URL) isn't cached.
const HEADER_LEN: usize = 1024;
const MAGIC: &str = "ruffle-aqw-cache 1";
/// Responses that stay fresh for less than this aren't worth a file.
const MIN_MAX_AGE: u64 = 60 * 60;
/// No entry is trusted without a request for longer than this, whatever the server says.
const MAX_MAX_AGE: u64 = 31 * 24 * 60 * 60;
/// Bodies larger than this aren't cached (the game's largest file is ~11 MB).
pub const MAX_BODY: u64 = 64 * 1024 * 1024;
/// The folder is trimmed again each time this much has been written.
const TRIM_EVERY: u64 = 64 * 1024 * 1024;
/// A hit marks its entry as used at most this often (the mark is the file's modification time).
const TOUCH_EVERY: Duration = Duration::from_secs(24 * 60 * 60);

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// What a response's headers say about keeping it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Policy {
    /// Seconds from now the body may be used without asking again.
    pub max_age: u64,
    pub etag: String,
    pub last_modified: String,
    pub content_type: String,
    /// `Content-Length`, when the server gave one.
    pub len: Option<u64>,
}

fn header_str(headers: &HeaderMap, name: header::HeaderName) -> &str {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
}

/// The `max-age` of a `Cache-Control` value, or `None` when it forbids keeping the response.
fn max_age(cache_control: &str) -> Option<u64> {
    let mut max_age = None;
    for directive in cache_control.split(',') {
        let directive = directive.trim().to_ascii_lowercase();
        match directive.as_str() {
            "no-store" | "no-cache" | "private" => return None,
            _ => {}
        }
        if let Some(seconds) = directive.strip_prefix("max-age=") {
            max_age = seconds.trim_matches('"').parse::<u64>().ok();
        }
    }
    max_age
}

impl Policy {
    /// The policy of a `200` response, or `None` when it must not be cached.
    pub fn of_response(headers: &HeaderMap) -> Option<Self> {
        let policy = Self::of_revalidation(headers)?;
        if headers.contains_key(header::CONTENT_RANGE) {
            return None;
        }
        // A private cache may keep a response that sets a cookie (RFC 9111, 3), but only one the
        // server calls `public` is a file meant for everyone rather than this session's answer.
        let public = header_str(headers, header::CACHE_CONTROL)
            .split(',')
            .any(|directive| directive.trim().eq_ignore_ascii_case("public"));
        if headers.contains_key(header::SET_COOKIE) && !public {
            return None;
        }
        let vary = header_str(headers, header::VARY);
        if !vary
            .split(',')
            .map(str::trim)
            .all(|name| name.is_empty() || name.eq_ignore_ascii_case("accept-encoding"))
        {
            return None;
        }
        // Without a validator a stale entry could only be fetched again, and nothing would
        // tell an updated file from the one on disk when the query changes.
        if policy.etag.is_empty() && policy.last_modified.is_empty() {
            return None;
        }
        if policy.len.is_some_and(|len| len > MAX_BODY) {
            return None;
        }
        Some(policy)
    }

    /// The policy a `304` response gives an entry that is already cached (a `304` needn't
    /// repeat the validators; the entry keeps its own then), or `None` when it doesn't say the
    /// entry may be used without asking: the entry then stays as it is, to be checked again.
    pub fn of_revalidation(headers: &HeaderMap) -> Option<Self> {
        let max_age = max_age(header_str(headers, header::CACHE_CONTROL))?;
        if max_age < MIN_MAX_AGE {
            return None;
        }
        // A shared cache in front of the server (Cloudflare) has already held it for `Age`.
        let age = header_str(headers, header::AGE).parse::<u64>().unwrap_or(0);
        let single_line = |name| {
            let value = header_str(headers, name);
            if value.contains(['\n', '\r']) {
                ""
            } else {
                value
            }
            .to_string()
        };
        Some(Self {
            max_age: max_age.min(MAX_MAX_AGE).saturating_sub(age),
            etag: single_line(header::ETAG),
            last_modified: single_line(header::LAST_MODIFIED),
            content_type: single_line(header::CONTENT_TYPE),
            len: header_str(headers, header::CONTENT_LENGTH).parse().ok(),
        })
    }
}

/// An entry's header: what it is and when it has to be checked again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Meta {
    /// The URL without its query and fragment.
    pub url: String,
    /// The query it was last fetched or revalidated with.
    pub query: String,
    pub etag: String,
    pub last_modified: String,
    pub content_type: String,
    /// When it was fetched or last revalidated (Unix seconds).
    pub stored: u64,
    pub max_age: u64,
    /// The body's length.
    pub len: u64,
}

impl Meta {
    fn encode(&self) -> Option<Vec<u8>> {
        let fields = [
            &self.url,
            &self.query,
            &self.etag,
            &self.last_modified,
            &self.content_type,
        ];
        if fields.iter().any(|field| field.contains(['\n', '\r'])) {
            return None;
        }
        let mut text = format!(
            "{MAGIC}\nurl={}\nquery={}\netag={}\nlast-modified={}\ncontent-type={}\nstored={}\nmax-age={}\nlen={}\n",
            self.url,
            self.query,
            self.etag,
            self.last_modified,
            self.content_type,
            self.stored,
            self.max_age,
            self.len
        )
        .into_bytes();
        if text.len() >= HEADER_LEN {
            return None;
        }
        text.resize(HEADER_LEN - 1, b' ');
        text.push(b'\n');
        Some(text)
    }

    fn decode(header: &[u8]) -> Option<Self> {
        let text = std::str::from_utf8(header).ok()?;
        let mut lines = text.split('\n');
        if lines.next()? != MAGIC {
            return None;
        }
        let mut field = |name: &str| -> Option<String> {
            let line = lines.next()?;
            Some(line.strip_prefix(name)?.strip_prefix('=')?.to_string())
        };
        Some(Self {
            url: field("url")?,
            query: field("query")?,
            etag: field("etag")?,
            last_modified: field("last-modified")?,
            content_type: field("content-type")?,
            stored: field("stored")?.parse().ok()?,
            max_age: field("max-age")?.parse().ok()?,
            len: field("len")?.parse().ok()?,
        })
    }

    /// Whether the body can answer a request for `url` without asking the server.
    pub fn is_fresh(&self, url: &Url) -> bool {
        let now = now();
        self.query == url.query().unwrap_or("")
            // A clock set back mustn't keep an entry fresh for ever.
            && now >= self.stored
            && now - self.stored < self.max_age
    }
}

/// A cached response.
pub struct Entry {
    pub meta: Meta,
    pub body: Vec<u8>,
}

/// The cache's key: `url` without its query and fragment.
fn key_url(url: &Url) -> String {
    let mut key = url.clone();
    key.set_query(None);
    key.set_fragment(None);
    key.into()
}

/// FNV-1a: the same file name for a URL in every build and process (std's hasher promises
/// neither). A collision only makes two URLs take turns in one entry: the header has the URL.
fn fnv64(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// What the cache has done since the process started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Requests answered from disk without a request to the server.
    pub hits: u64,
    /// Requests answered from disk after the server said the entry is still current.
    pub revalidated: u64,
    /// Responses downloaded and written to the cache.
    pub stored: u64,
    /// Body bytes that didn't have to be downloaded.
    pub bytes_saved: u64,
}

/// The disk cache. One per process is enough: every player's navigator can share it, and so
/// can other processes using the same folder.
pub struct HttpCache {
    dir: PathBuf,
    max_bytes: u64,
    /// Bytes written since the folder was last trimmed.
    written: AtomicU64,
    /// Makes temporary file names unique within the process.
    serial: AtomicU64,
    hits: AtomicU64,
    revalidated: AtomicU64,
    stored: AtomicU64,
    bytes_saved: AtomicU64,
}

impl HttpCache {
    /// A cache in `dir` (created if missing) that keeps at most about `max_bytes` of files.
    /// Trims the folder once, on a thread of its own.
    pub fn new(dir: impl Into<PathBuf>, max_bytes: u64) -> io::Result<Arc<Self>> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        let cache = Arc::new(Self {
            dir,
            max_bytes,
            written: AtomicU64::new(0),
            serial: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            revalidated: AtomicU64::new(0),
            stored: AtomicU64::new(0),
            bytes_saved: AtomicU64::new(0),
        });
        let trimmer = cache.clone();
        std::thread::Builder::new()
            .name("HTTP cache trim".to_string())
            .spawn(move || trimmer.trim())?;
        Ok(cache)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            revalidated: self.revalidated.load(Ordering::Relaxed),
            stored: self.stored.load(Ordering::Relaxed),
            bytes_saved: self.bytes_saved.load(Ordering::Relaxed),
        }
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{:016x}.entry", fnv64(key)))
    }

    /// The entry for `url`, fresh or not. Reads the whole file: call it off the player's thread.
    pub fn lookup(&self, url: &Url) -> Option<Entry> {
        let key = key_url(url);
        let path = self.path(&key);
        let mut file = File::open(&path).ok()?;
        let mut header = [0u8; HEADER_LEN];
        file.read_exact(&mut header).ok()?;
        let meta = Meta::decode(&header).filter(|meta| meta.url == key)?;
        if meta.len > MAX_BODY {
            return None;
        }
        let mut body = Vec::with_capacity(meta.len as usize);
        file.read_to_end(&mut body).ok()?;
        if body.len() as u64 != meta.len {
            // Cut short (a full disk, a crash): as good as missing.
            return None;
        }
        // The modification time is "last used", which trimming goes by.
        let stale_mark = file
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age > TOUCH_EVERY);
        if stale_mark {
            let _ = File::options()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_modified(SystemTime::now()));
        }
        Some(Entry { meta, body })
    }

    /// Counts a request that `entry` answered; `asked` is whether the server was asked first.
    pub fn count_served(&self, entry: &Entry, asked: bool) {
        let counter = if asked { &self.revalidated } else { &self.hits };
        counter.fetch_add(1, Ordering::Relaxed);
        self.bytes_saved
            .fetch_add(entry.meta.len, Ordering::Relaxed);
    }

    /// Keeps `body`, the response to `url` whose headers gave `policy`.
    pub fn store(&self, url: &Url, policy: &Policy, body: &[u8]) {
        let meta = Meta {
            url: key_url(url),
            query: url.query().unwrap_or("").to_string(),
            etag: policy.etag.clone(),
            last_modified: policy.last_modified.clone(),
            content_type: policy.content_type.clone(),
            stored: now(),
            max_age: policy.max_age,
            len: body.len() as u64,
        };
        let Some(header) = meta.encode() else {
            return;
        };
        let path = self.path(&meta.url);
        let temporary = self.dir.join(format!(
            "{:016x}.{}-{}.tmp",
            fnv64(&meta.url),
            std::process::id(),
            self.serial.fetch_add(1, Ordering::Relaxed)
        ));
        let write = || -> io::Result<()> {
            let mut file = File::create(&temporary)?;
            file.write_all(&header)?;
            file.write_all(body)?;
            drop(file);
            fs::rename(&temporary, &path)
        };
        match write() {
            Ok(()) => {
                self.stored.fetch_add(1, Ordering::Relaxed);
                let written = (HEADER_LEN + body.len()) as u64;
                if self.written.fetch_add(written, Ordering::Relaxed) + written >= TRIM_EVERY {
                    self.written.store(0, Ordering::Relaxed);
                    self.trim();
                }
            }
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                tracing::warn!("Couldn't cache {}: {error}", meta.url);
            }
        }
    }

    /// The server confirmed `meta`'s entry for `url` (a `304` with these headers): it is fresh
    /// again, for this query. Rewrites the header in place.
    pub fn refresh(&self, url: &Url, meta: &Meta, policy: &Policy) {
        let path = self.path(&meta.url);
        let or_old = |new: &String, old: &String| if new.is_empty() { old } else { new }.clone();
        let meta = Meta {
            query: url.query().unwrap_or("").to_string(),
            etag: or_old(&policy.etag, &meta.etag),
            last_modified: or_old(&policy.last_modified, &meta.last_modified),
            stored: now(),
            max_age: policy.max_age,
            ..meta.clone()
        };
        let Some(header) = meta.encode() else {
            return;
        };
        let rewrite = || -> io::Result<()> {
            let mut file = File::options().read(true).write(true).open(&path)?;
            // Another client may have replaced the entry since it was read.
            let mut current = [0u8; HEADER_LEN];
            file.read_exact(&mut current)?;
            let same = Meta::decode(&current)
                .is_some_and(|current| current.url == meta.url && current.len == meta.len);
            if same {
                file.seek(SeekFrom::Start(0))?;
                file.write_all(&header)?;
            }
            Ok(())
        };
        if let Err(error) = rewrite() {
            tracing::debug!("Couldn't refresh the cached {}: {error}", meta.url);
        }
    }

    /// Removes the entries used longest ago until the folder is under 80% of `max_bytes`, and
    /// temporary files a crashed client left behind.
    fn trim(&self) {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return;
        };
        let mut files = Vec::new();
        let mut total = 0u64;
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
            match path.extension().and_then(|extension| extension.to_str()) {
                Some("entry") => {
                    total += metadata.len();
                    files.push((modified, metadata.len(), path));
                }
                Some("tmp") => {
                    let abandoned = modified
                        .elapsed()
                        .is_ok_and(|age| age > Duration::from_secs(60 * 60));
                    if abandoned {
                        let _ = fs::remove_file(path);
                    }
                }
                _ => {}
            }
        }
        if total <= self.max_bytes {
            return;
        }
        files.sort();
        let target = self.max_bytes / 5 * 4;
        let mut removed = 0;
        for (_, len, path) in files {
            if total <= target {
                break;
            }
            if fs::remove_file(path).is_ok() {
                total -= len;
                removed += 1;
            }
        }
        tracing::info!(
            "HTTP cache: removed {removed} files used longest ago, {} MB left",
            total / (1024 * 1024)
        );
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.append(*name, value.parse().unwrap());
        }
        map
    }

    /// A folder of this test's own, removed when dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "ruffle-aqw-cache-{name}-{}-{}",
                std::process::id(),
                now()
            ));
            let _ = fs::remove_dir_all(&dir);
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const AQW: &[(&str, &str)] = &[
        ("cache-control", "public, max-age=2678400"),
        ("etag", "\"80161a8cc3f0cb1:0\""),
        ("last-modified", "Fri, 01 Apr 2011 23:21:37 GMT"),
        ("content-length", "5"),
        ("age", "1000"),
    ];

    #[test]
    fn game_files_are_cacheable() {
        let policy = Policy::of_response(&headers(AQW)).unwrap();
        assert_eq!(policy.max_age, 2678400 - 1000);
        assert_eq!(policy.etag, "\"80161a8cc3f0cb1:0\"");
        assert_eq!(policy.len, Some(5));
    }

    #[test]
    fn api_answers_are_not() {
        // game.aq.com/game/api/data/gameversion
        let api = headers(&[
            ("cache-control", "public, max-age=2"),
            ("vary", "*"),
            ("set-cookie", "__cflb=x; HttpOnly"),
        ]);
        assert_eq!(Policy::of_response(&api), None);
        for (name, value) in [
            ("cache-control", "no-store, max-age=99999"),
            ("cache-control", "private, max-age=99999"),
            ("cache-control", "no-cache"),
            ("vary", "Cookie"),
            ("content-range", "bytes 0-4/10"),
            ("content-length", "999999999999"),
        ] {
            let mut pairs = AQW.to_vec();
            if name == "cache-control" || name == "content-length" {
                pairs.retain(|(n, _)| *n != name);
            }
            pairs.push((name, value));
            assert_eq!(
                Policy::of_response(&headers(&pairs)),
                None,
                "{name}: {value}"
            );
        }
        // A cookie: kept only when the file is `public` (AQW's CDN sets Cloudflare's).
        let mut pairs = AQW.to_vec();
        pairs.push(("set-cookie", "__cflb=x; HttpOnly"));
        assert!(Policy::of_response(&headers(&pairs)).is_some());
        let cookie = [
            ("cache-control", "max-age=2678400"),
            ("etag", "\"a\""),
            ("set-cookie", "a=b"),
        ];
        assert_eq!(Policy::of_response(&headers(&cookie)), None);
        // No max-age at all, and nothing to revalidate with.
        assert_eq!(Policy::of_response(&headers(&[("etag", "\"a\"")])), None);
        assert_eq!(
            Policy::of_response(&headers(&[("cache-control", "max-age=99999")])),
            None
        );
        // `Vary: Accept-Encoding` is fine: the client always sends the same one.
        let mut pairs = AQW.to_vec();
        pairs.push(("vary", "Accept-Encoding"));
        assert!(Policy::of_response(&headers(&pairs)).is_some());
    }

    #[test]
    fn header_round_trip() {
        let meta = Meta {
            url: "https://game.aq.com/game/gamefiles/classes/F/Ranger.swf".to_string(),
            query: "v=5.32".to_string(),
            etag: "\"80161a8cc3f0cb1:0\"".to_string(),
            last_modified: "Fri, 01 Apr 2011 23:21:37 GMT".to_string(),
            content_type: "application/x-shockwave-flash".to_string(),
            stored: 1_794_000_000,
            max_age: 2678400,
            len: 16487,
        };
        let header = meta.encode().unwrap();
        assert_eq!(header.len(), HEADER_LEN);
        assert_eq!(Meta::decode(&header), Some(meta.clone()));
        // Too long for the header, or able to forge a field: not cached.
        let long = Meta {
            url: "x".repeat(HEADER_LEN),
            ..meta.clone()
        };
        assert_eq!(long.encode(), None);
        let forged = Meta {
            etag: "a\nlen=0".to_string(),
            ..meta
        };
        assert_eq!(forged.encode(), None);
        assert_eq!(Meta::decode(&[b' '; HEADER_LEN]), None);
    }

    #[test]
    fn store_lookup_refresh() {
        let dir = TempDir::new("store");
        let cache = HttpCache::new(&dir.0, 1 << 30).unwrap();
        let url = Url::parse("https://game.aq.com/game/gamefiles/mon/Frogzard.swf?v=5.32").unwrap();
        assert!(cache.lookup(&url).is_none());

        let policy = Policy::of_response(&headers(AQW)).unwrap();
        cache.store(&url, &policy, b"hello");
        let entry = cache.lookup(&url).unwrap();
        assert_eq!(entry.body, b"hello");
        assert!(entry.meta.is_fresh(&url));

        // Another query is the same entry, but has to be checked with the server.
        let updated =
            Url::parse("https://game.aq.com/game/gamefiles/mon/Frogzard.swf?v=5.33").unwrap();
        let entry = cache.lookup(&updated).unwrap();
        assert_eq!(entry.body, b"hello");
        assert!(!entry.meta.is_fresh(&updated));

        // The server says it hasn't changed: fresh for the new query, not the old one.
        cache.refresh(&updated, &entry.meta, &policy);
        let entry = cache.lookup(&updated).unwrap();
        assert_eq!(entry.body, b"hello");
        assert!(entry.meta.is_fresh(&updated));
        assert!(!entry.meta.is_fresh(&url));

        // A 304 without validators keeps the entry's own.
        let bare =
            Policy::of_revalidation(&headers(&[("cache-control", "max-age=86400")])).unwrap();
        cache.refresh(&updated, &entry.meta, &bare);
        let entry = cache.lookup(&updated).unwrap();
        assert_eq!(entry.meta.etag, "\"80161a8cc3f0cb1:0\"");
        assert_eq!(entry.meta.max_age, 86400);

        // Past its max-age: stale.
        let old = Meta {
            stored: now() - 10,
            max_age: 5,
            ..entry.meta.clone()
        };
        assert!(!old.is_fresh(&updated));
        // From the future (the clock was set back): stale.
        let future = Meta {
            stored: now() + 1000,
            ..entry.meta
        };
        assert!(!future.is_fresh(&updated));
    }

    #[test]
    fn damaged_entries_are_misses() {
        let dir = TempDir::new("damaged");
        let cache = HttpCache::new(&dir.0, 1 << 30).unwrap();
        let url = Url::parse("https://game.aq.com/game/gamefiles/a.swf").unwrap();
        let policy = Policy::of_response(&headers(AQW)).unwrap();
        cache.store(&url, &policy, b"hello");
        let path = cache.path(&key_url(&url));

        // Cut short.
        let whole = fs::read(&path).unwrap();
        fs::write(&path, &whole[..whole.len() - 2]).unwrap();
        assert!(cache.lookup(&url).is_none());
        // Another URL's entry under this name (a hash collision).
        fs::write(&path, &whole).unwrap();
        assert!(cache.lookup(&url).is_some());
        let other = String::from_utf8(whole).unwrap().replace("a.swf", "b.swf");
        fs::write(&path, other).unwrap();
        assert!(cache.lookup(&url).is_none());
        // Not an entry at all.
        fs::write(&path, b"junk").unwrap();
        assert!(cache.lookup(&url).is_none());
    }

    #[test]
    fn trims_the_entries_used_longest_ago() {
        let dir = TempDir::new("trim");
        // Room for about three of the entries below.
        let entry_len = (HEADER_LEN + 1000) as u64;
        let cache = HttpCache::new(&dir.0, entry_len * 3 + 100).unwrap();
        let policy = Policy::of_response(&headers(AQW)).unwrap();
        let url =
            |n: usize| Url::parse(&format!("https://game.aq.com/game/gamefiles/{n}.swf")).unwrap();
        for n in 0..6 {
            cache.store(&url(n), &policy, &[n as u8; 1000]);
            // 0 was used longest ago, 5 last.
            let used = SystemTime::now() - Duration::from_secs(600 - n as u64 * 60);
            File::options()
                .write(true)
                .open(cache.path(&key_url(&url(n))))
                .unwrap()
                .set_modified(used)
                .unwrap();
        }
        fs::write(dir.0.join("0000000000000000.1-1.tmp"), b"in progress").unwrap();
        cache.trim();
        let kept: Vec<usize> = (0..6)
            .filter(|n| cache.lookup(&url(*n)).is_some())
            .collect();
        assert_eq!(kept, [4, 5]);
        // A temporary file that's being written is left alone.
        assert!(dir.0.join("0000000000000000.1-1.tmp").exists());
    }
}
