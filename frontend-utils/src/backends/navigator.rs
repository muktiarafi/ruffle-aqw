mod cache;
mod fetch;

pub use crate::backends::navigator::cache::{CacheStats, HttpCache};
use crate::backends::navigator::cache::{CacheWriter, Policy};
use crate::backends::navigator::fetch::{Response, ResponseBody};
use crate::content::PlayingContent;
use async_channel::{Receiver, Sender, TryRecvError};
use async_io::Timer;
use futures_lite::FutureExt;
use reqwest::{Proxy, cookie, header};
use ruffle_core::backend::navigator::{
    ErrorResponse, NavigationMethod, NavigatorBackend, OwnedFuture, Request, SocketMode,
    SuccessResponse, async_return, create_fetch_error, get_encoding,
};
use ruffle_core::indexmap::IndexMap;
use ruffle_core::loader::Error;
use ruffle_core::socket::{ConnectionState, SocketAction, SocketHandle};
use std::collections::HashSet;
use std::fs::File;
use std::future::Future;
use std::io;
use std::io::ErrorKind;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::warn;
use url::{ParseError, Url};

pub trait NavigatorInterface: Clone + Send + 'static {
    fn navigate_to_website(&self, url: Url);

    fn open_file(&self, path: &Path) -> impl std::future::Future<Output = io::Result<File>> + Send;

    fn confirm_socket(
        &self,
        host: &str,
        port: u16,
    ) -> impl std::future::Future<Output = bool> + Send;
}

pub trait FutureSpawner<Err> {
    fn spawn(&self, future: OwnedFuture<(), Err>);
}

/// Implementation of `NavigatorBackend` for non-web environments that can call
/// out to a web browser.
pub struct ExternalNavigatorBackend<F: FutureSpawner<Error>, I: NavigatorInterface> {
    /// Sink for tasks sent to us through `spawn_future`.
    future_spawner: F,

    /// The url to use for all relative fetches.
    base_url: Url,

    // Client to use for network requests
    client: Option<Rc<reqwest::Client>>,

    socket_allowed: HashSet<String>,

    socket_mode: SocketMode,

    upgrade_to_https: bool,

    content: Rc<PlayingContent>,

    interface: I,

    /// AQW build: the disk cache for cacheable downloads (see `cache.rs`); `None`: every
    /// request goes to the network, as in Ruffle.
    http_cache: Option<Arc<HttpCache>>,
}

impl<F: FutureSpawner<Error>, I: NavigatorInterface> ExternalNavigatorBackend<F, I> {
    /// Construct a navigator backend with fetch and async capability.
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        mut base_url: Url,
        referer: Option<Url>,
        cookie: Option<String>,
        future_spawner: F,
        proxy: Option<Url>,
        upgrade_to_https: bool,
        socket_allowed: HashSet<String>,
        socket_mode: SocketMode,
        content: Rc<PlayingContent>,
        interface: I,
    ) -> Self {
        let mut builder = reqwest::ClientBuilder::new()
            .cookie_store(true)
            .user_agent(concat!(
                "Ruffle/",
                env!("CARGO_PKG_VERSION"),
                " (https://ruffle.rs)"
            ));

        if let Some(referer) = referer {
            let mut headers = header::HeaderMap::new();
            headers.insert(header::REFERER, referer.to_string().parse().unwrap());
            builder = builder.default_headers(headers);
        }

        if let Some(cookie) = cookie {
            let cookie_jar = cookie::Jar::default();
            cookie_jar.add_cookie_str(&cookie, &base_url);
            let cookie_store = std::sync::Arc::new(cookie_jar);
            builder = builder.cookie_provider(cookie_store)
        }

        if let Some(proxy) = proxy {
            match Proxy::all(proxy.clone()) {
                Ok(proxy) => {
                    builder = builder.proxy(proxy);
                }
                Err(e) => {
                    tracing::error!("Couldn't configure proxy {proxy}: {e}")
                }
            }
        }

        let client = builder.build().ok().map(Rc::new);

        // Force replace the last segment with empty. //

        if let Ok(mut base_url) = base_url.path_segments_mut() {
            base_url.pop().pop_if_empty().push("");
        }

        Self {
            future_spawner,
            client,
            base_url,
            upgrade_to_https,
            socket_allowed,
            socket_mode,
            content,
            interface,
            http_cache: None,
        }
    }

    /// AQW build: answers cacheable downloads from `cache`, and keeps them there.
    pub fn with_http_cache(mut self, cache: Option<Arc<HttpCache>>) -> Self {
        self.http_cache = cache;
        self
    }
}

impl<F: FutureSpawner<Error> + 'static, I: NavigatorInterface> NavigatorBackend
    for ExternalNavigatorBackend<F, I>
{
    fn navigate_to_url(
        &self,
        url: &str,
        _target: &str,
        vars_method: Option<(NavigationMethod, IndexMap<String, String>)>,
    ) {
        //TODO: Should we return a result for failed opens? Does Flash care?

        //NOTE: Flash desktop players / projectors ignore the window parameter,
        //      unless it's a `_layer`, and we shouldn't handle that anyway.
        let mut parsed_url = match self.resolve_url(url) {
            Ok(parsed_url) => parsed_url,
            Err(e) => {
                tracing::error!(
                    "Could not parse URL because of {}, the corrupt URL was: {}",
                    e,
                    url
                );
                return;
            }
        };

        let modified_url = match vars_method {
            Some((_, query_pairs)) if !query_pairs.is_empty() => {
                {
                    //lifetime limiter because we don't have NLL yet
                    let mut modifier = parsed_url.query_pairs_mut();

                    for (k, v) in query_pairs.iter() {
                        modifier.append_pair(k, v);
                    }
                }

                parsed_url
            }
            _ => parsed_url,
        };

        if modified_url.scheme() == "javascript" {
            tracing::warn!(
                "SWF tried to run a script on desktop, but javascript calls are not allowed"
            );
            return;
        }

        self.interface.navigate_to_website(modified_url);
    }

    fn fetch(&self, request: Request) -> OwnedFuture<Box<dyn SuccessResponse>, ErrorResponse> {
        // TODO: honor sandbox type (local-with-filesystem, local-with-network, remote, ...)
        let mut processed_url = match self.resolve_url(request.url()) {
            Ok(url) => url,
            Err(e) => {
                return async_return(Err(create_fetch_error(request.url(), e)));
            }
        };

        let client = self.client.clone();
        // AQW build: only a plain GET can be answered from the cache or kept in it.
        let http_cache = self.http_cache.clone().filter(|_| {
            request.method() == NavigationMethod::Get
                && request.body().is_none()
                && request.headers().is_empty()
        });

        match processed_url.scheme() {
            "file" => {
                let content = self.content.clone();
                let interface = self.interface.clone();
                Box::pin(async move {
                    // We send the original url (including query parameters)
                    // back to ruffle_core in the `Response`
                    let response_url = processed_url.clone();
                    // Flash supports query parameters with local urls.
                    // SwfMovie takes care of exposing those to ActionScript -
                    // when we actually load a filesystem url, strip them out.
                    processed_url.set_query(None);

                    let contents = content.get_local_file(&processed_url, interface).await;

                    let response: Box<dyn SuccessResponse> = Box::new(Response {
                        url: response_url.to_string(),
                        response_body: ResponseBody::File(contents),
                        text_encoding: None,
                        status: 0,
                        redirected: false,
                        cache: None,
                    });

                    Ok(response)
                })
            }
            _ => Box::pin(async move {
                let client = client.ok_or_else(|| ErrorResponse {
                    url: processed_url.to_string(),
                    error: Error::FetchError("Network unavailable".to_string()),
                })?;

                // AQW build: the disk cache. A fresh entry is the answer; a stale one (or one
                // fetched with another query string) is checked with the server below.
                let cached = match &http_cache {
                    Some(cache) => {
                        let (cache, url) = (cache.clone(), processed_url.clone());
                        spawn_tokio(async move {
                            tokio::task::spawn_blocking(move || cache.lookup(&url))
                                .await
                                .ok()
                                .flatten()
                        })
                        .await
                    }
                    None => None,
                };
                let from_cache = |entry: cache::Entry| -> Box<dyn SuccessResponse> {
                    Box::new(Response {
                        url: processed_url.to_string(),
                        text_encoding: get_encoding(&entry.meta.content_type),
                        response_body: ResponseBody::File(Ok(entry.body)),
                        status: 200,
                        redirected: false,
                        cache: None,
                    })
                };
                let cached = match (cached, &http_cache) {
                    (Some(entry), Some(cache)) if entry.meta.is_fresh(&processed_url) => {
                        cache.count_served(&entry, false);
                        return Ok(from_cache(entry));
                    }
                    (cached, _) => cached,
                };

                let build_request = || {
                    let mut request_builder = match request.method() {
                        NavigationMethod::Get => client.get(processed_url.clone()),
                        NavigationMethod::Post => client.post(processed_url.clone()),
                    };
                    let (body_data, mime) = request.body().clone().unwrap_or_default();
                    for (name, val) in request.headers().iter() {
                        request_builder = request_builder.header(name, val);
                    }
                    if let Some(entry) = &cached {
                        if !entry.meta.etag.is_empty() {
                            request_builder =
                                request_builder.header(header::IF_NONE_MATCH, &entry.meta.etag);
                        }
                        if !entry.meta.last_modified.is_empty() {
                            request_builder = request_builder
                                .header(header::IF_MODIFIED_SINCE, &entry.meta.last_modified);
                        }
                    }
                    request_builder
                        .header("Content-Type", &mime)
                        .body(body_data)
                };

                // AQW build (macOS): a DNS or connection hiccup while the game starts used to
                // leave the window blank until Mugetsu was restarted. Retry a GET once; POSTs
                // are not retried, as repeating them could repeat an action (e.g. a login).
                #[cfg(target_os = "macos")]
                let response = {
                    let mut attempt = 0;
                    loop {
                        match spawn_tokio(build_request().send()).await {
                            Err(e)
                                if attempt == 0
                                    && request.method() == NavigationMethod::Get
                                    && e.is_connect() =>
                            {
                                attempt = 1;
                                tracing::warn!(
                                    "Could not connect to {processed_url}: {e}; retrying once"
                                );
                                Timer::after(Duration::from_millis(750)).await;
                            }
                            response => break response,
                        }
                    }
                };
                #[cfg(not(target_os = "macos"))]
                let response = spawn_tokio(build_request().send()).await;

                // AQW build: with the network down, a cached file is better than none.
                let response = match (response, cached, &http_cache) {
                    (Err(e), Some(entry), Some(cache)) => {
                        tracing::warn!(
                            "Couldn't check {processed_url} with the server ({e}); using the cached file"
                        );
                        cache.count_served(&entry, true);
                        return Ok(from_cache(entry));
                    }
                    (Ok(response), Some(entry), Some(cache))
                        if response.status() == reqwest::StatusCode::NOT_MODIFIED =>
                    {
                        // Unchanged: fresh again, for this query string.
                        if let Some(policy) = Policy::of_revalidation(response.headers()) {
                            let (cache, url, meta) =
                                (cache.clone(), processed_url.clone(), entry.meta.clone());
                            tokio::task::spawn_blocking(move || {
                                cache.refresh(&url, &meta, &policy)
                            });
                        }
                        cache.count_served(&entry, true);
                        return Ok(from_cache(entry));
                    }
                    (response, _, _) => response,
                };

                let response = response.map_err(|e| {
                    let inner = if e.is_connect() {
                        Error::InvalidDomain(processed_url.to_string())
                    } else {
                        Error::FetchError(e.to_string())
                    };
                    ErrorResponse {
                        url: processed_url.to_string(),
                        error: inner,
                    }
                })?;

                let url = response.url().to_string();
                let text_encoding = response
                    .headers()
                    .get("Content-Type")
                    .and_then(|content_type| content_type.to_str().ok())
                    .and_then(get_encoding);
                let status = response.status().as_u16();
                let redirected = *response.url() != processed_url;
                if !response.status().is_success() {
                    let error = Error::HttpNotOk(
                        format!("Got {}", response.status()),
                        status,
                        redirected,
                        response.content_length().unwrap_or_default(),
                    );
                    return Err(ErrorResponse { url, error });
                }

                // AQW build: keep what the server allows keeping, as its body is read.
                let cache = http_cache
                    .filter(|_| status == 200 && !redirected)
                    .and_then(|cache| {
                        let policy = Policy::of_response(response.headers())?;
                        let writer = CacheWriter::new(cache, processed_url.clone(), policy);
                        Some(Arc::new(Mutex::new(Some(writer))))
                    });

                let response: Box<dyn SuccessResponse> = Box::new(Response {
                    url,
                    response_body: ResponseBody::Network(Arc::new(Mutex::new(Some(response)))),
                    text_encoding,
                    status,
                    redirected,
                    cache,
                });
                Ok(response)
            }),
        }
    }

    fn resolve_url(&self, url: &str) -> Result<Url, ParseError> {
        match self.base_url.join(url) {
            Ok(url) => Ok(self.pre_process_url(url)),
            Err(error) => Err(error),
        }
    }

    fn spawn_future(&mut self, future: OwnedFuture<(), Error>) {
        self.future_spawner.spawn(future);
    }

    fn pre_process_url(&self, mut url: Url) -> Url {
        if self.upgrade_to_https && url.scheme() == "http" && url.set_scheme("https").is_err() {
            tracing::error!("Url::set_scheme failed on: {}", url);
        }
        url
    }

    fn connect_socket(
        &mut self,
        host: String,
        port: u16,
        timeout: Duration,
        handle: SocketHandle,
        receiver: Receiver<Vec<u8>>,
        sender: Sender<SocketAction>,
    ) {
        /// Tries to send the given action properly handling failures.
        ///
        /// Returns `true` when the action has been sent properly,
        /// `false` when the channel is closed.
        async fn send_action(sender: &Sender<SocketAction>, action: SocketAction) -> bool {
            sender
                .send(action)
                .await
                .inspect_err(|err| tracing::warn!("Failed to send SocketAction: {}", err))
                .is_ok()
        }

        let addr = format!("{host}:{port}");
        let is_allowed = self.socket_allowed.contains(&addr);
        let socket_mode = self.socket_mode;
        let interface = self.interface.clone();

        let future = Box::pin(async move {
            match (is_allowed, socket_mode) {
                (false, SocketMode::Allow) | (true, _) => {} // the process is allowed to continue. just dont do anything.
                (false, SocketMode::Deny) => {
                    // Just fail the connection.
                    let action = SocketAction::Connect(handle, ConnectionState::Failed);
                    let _ = send_action(&sender, action).await;

                    tracing::warn!(
                        "SWF tried to open a socket, but opening a socket is not allowed"
                    );

                    return;
                }
                (false, SocketMode::Ask) => {
                    let attempt_sandbox_connect = interface.confirm_socket(&host, port).await;

                    if !attempt_sandbox_connect {
                        // fail the connection.
                        let action = SocketAction::Connect(handle, ConnectionState::Failed);
                        let _ = send_action(&sender, action).await;
                        return;
                    }
                }
            }

            let host2 = host.clone();

            let timeout = async {
                Timer::after(timeout).await;
                Result::<TcpStream, io::Error>::Err(io::Error::new(ErrorKind::TimedOut, ""))
            };

            let mut stream = match TcpStream::connect((host, port)).or(timeout).await {
                Err(e) if e.kind() == ErrorKind::TimedOut => {
                    warn!("Connection to {}:{} timed out", host2, port);
                    let action = SocketAction::Connect(handle, ConnectionState::TimedOut);
                    let _ = send_action(&sender, action).await;
                    return;
                }
                Ok(stream) => {
                    let action = SocketAction::Connect(handle, ConnectionState::Connected);
                    if !send_action(&sender, action).await {
                        return;
                    }
                    stream
                }
                Err(err) => {
                    warn!("Failed to connect to {}:{}, error: {}", host2, port, err);
                    let action = SocketAction::Connect(handle, ConnectionState::Failed);
                    let _ = send_action(&sender, action).await;
                    return;
                }
            };

            let sender = sender;
            //NOTE: We clone the sender here as we cant share it between async tasks.
            let sender2 = sender.clone();
            let (mut read, mut write) = stream.split();

            let read = async move {
                loop {
                    let mut buffer = [0; 4096];

                    match read.read(&mut buffer).await {
                        Err(e) if e.kind() == ErrorKind::TimedOut => {} // try again later.
                        Err(_) | Ok(0) => {
                            let _ = send_action(&sender, SocketAction::Close(handle)).await;
                            break;
                        }
                        Ok(read) => {
                            let buffer = buffer.into_iter().take(read).collect::<Vec<_>>();

                            let action = SocketAction::Data(handle, buffer);
                            if !send_action(&sender, action).await {
                                return;
                            }
                        }
                    };
                }
            };

            let write = async move {
                let mut pending_write = vec![];

                loop {
                    let close_connection = loop {
                        match receiver.try_recv() {
                            Ok(val) => {
                                pending_write.extend(val);
                            }
                            Err(TryRecvError::Empty) => break false,
                            Err(TryRecvError::Closed) => {
                                //NOTE: Channel sender has been dropped.
                                //      This means we have to close the connection,
                                //      but not here, as we might have a pending write.
                                break true;
                            }
                        }
                    };

                    if !pending_write.is_empty() {
                        match write.write(&pending_write).await {
                            Err(e) if e.kind() == ErrorKind::TimedOut => {} // try again later.
                            Err(_) => {
                                let _ = send_action(&sender2, SocketAction::Close(handle)).await;
                                return;
                            }
                            Ok(written) => {
                                let _ = pending_write.drain(..written);
                            }
                        }
                    } else if close_connection {
                        return;
                    } else {
                        // Receiver is empty and there's no pending data,
                        // we may block here and wait for new data.
                        match receiver.recv().await {
                            Ok(val) => {
                                pending_write.extend(val);
                            }
                            Err(_) => {
                                // Ignore the error here, it will be
                                // reported again in try_recv.
                            }
                        }
                    }
                }
            };

            //NOTE: If one future exits, this will take the other one down too.
            tokio::select! {
               _ = read => {},
               _ = write => {},
            }

            if let Err(e) = stream.shutdown().await {
                tracing::warn!("Failed to shutdown write half of TcpStream: {e}");
            }
        });

        tokio::spawn(future);
    }
}

/// Spawns a new asynchronous task in a tokio runtime, without the current executor needing to belong to tokio
pub async fn spawn_tokio<F>(future: F) -> F::Output
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let (sender, receiver) = tokio::sync::oneshot::channel();
    tokio::spawn(async move { sender.send(future.await) });
    tokio::task::unconstrained(receiver)
        .await
        .expect("Oneshot should succeed")
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use ruffle_core::socket::SocketAction::{Close, Connect, Data};
    use std::net::SocketAddr;
    use std::str::FromStr;
    use tokio::net::TcpListener;
    use tokio::task;

    use crate::content::ContentDescriptor;

    use super::*;

    impl NavigatorInterface for () {
        fn navigate_to_website(&self, _url: Url) {}

        async fn open_file(&self, path: &Path) -> io::Result<File> {
            File::open(path)
        }

        async fn confirm_socket(&self, _host: &str, _port: u16) -> bool {
            true
        }
    }

    const TIMEOUT_ZERO: Duration = Duration::ZERO;
    // The timeout has to be large enough to allow "instantaneous" actions
    // and local IO to execute, but small enough to fail tests quickly.
    const TIMEOUT: Duration = Duration::from_secs(1);

    struct TestFutureSpawner;

    impl<E: 'static> FutureSpawner<E> for TestFutureSpawner {
        fn spawn(&self, future: OwnedFuture<(), E>) {
            task::spawn_local(future);
        }
    }

    macro_rules! async_timeout {
        () => {
            async {
                Timer::after(TIMEOUT).await;
                panic!("An action which should complete timed out")
            }
        };
    }

    macro_rules! async_test {
        (
            async fn $test_name:ident() $content:block
        ) => {
            #[tokio::test(flavor = "current_thread")]
            async fn $test_name() {
                task::LocalSet::new().run_until(async move $content).await;
            }
        }
    }

    macro_rules! dummy_handle {
        () => {
            SocketHandle::default()
        };
    }

    macro_rules! assert_next_socket_actions {
        ($receiver:expr;) => {
            // no more actions
        };
        ($receiver:expr; $action:expr, $($more:expr,)*) => {
            assert_eq!($receiver.recv().or(async_timeout!()).await.expect("receive action"), $action);
            assert_next_socket_actions!($receiver; $($more,)*);
        };
    }

    fn new_test_backend(socket_allow: bool) -> ExternalNavigatorBackend<TestFutureSpawner, ()> {
        let url = Url::parse("https://example.com/path/").unwrap();
        ExternalNavigatorBackend::new(
            url.clone(),
            None,
            None,
            TestFutureSpawner,
            None,
            false,
            Default::default(),
            if socket_allow {
                SocketMode::Allow
            } else {
                SocketMode::Deny
            },
            Rc::new(PlayingContent::DirectFile(ContentDescriptor {
                url,
                root_content_path: None,
            })),
            (),
        )
    }

    async fn start_test_server() -> (task::JoinHandle<TcpStream>, SocketAddr) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let accept_task = task::spawn_local(async move {
            let (socket, _) = listener.accept().or(async_timeout!()).await.unwrap();
            socket
        });
        (accept_task, addr)
    }

    fn connect_test_socket(
        addr: SocketAddr,
        timeout: Duration,
        socket_allow: bool,
    ) -> (Sender<Vec<u8>>, Receiver<SocketAction>) {
        let mut backend = new_test_backend(socket_allow);

        let (write, receiver) = async_channel::unbounded();
        let (sender, read) = async_channel::unbounded();

        backend.connect_socket(
            addr.ip().to_string(),
            addr.port(),
            timeout,
            dummy_handle!(),
            receiver,
            sender,
        );

        (write, read)
    }

    async fn write_server(server_socket: &mut TcpStream, data: &str) {
        server_socket
            .write(data.as_bytes())
            .or(async_timeout!())
            .await
            .expect("server write");
    }

    async fn read_server(server_socket: &mut TcpStream) -> String {
        let mut buffer = [0; 4096];

        let read = match server_socket.read(&mut buffer).await {
            Err(e) => panic!("server read error: {e}"),
            Ok(read) => read,
        };

        let buffer = buffer.into_iter().take(read).collect::<Vec<_>>();
        String::from_utf8(buffer).unwrap()
    }

    async fn write_client(client_write: &Sender<Vec<u8>>, data: &str) {
        client_write
            .send(data.as_bytes().to_vec())
            .or(async_timeout!())
            .await
            .expect("client write");
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_timeout() {
        let (_accept_task, addr) = start_test_server().await;
        let (_client_write, client_read) = connect_test_socket(addr, TIMEOUT_ZERO, true);
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::TimedOut),
        );
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_connect() {
        let (accept_task, addr) = start_test_server().await;
        let (_client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);
        let _server_socket = accept_task.await.unwrap();
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Connected),
        );
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_deny() {
        let (_accept_task, addr) = start_test_server().await;
        let (_client_write, client_read) = connect_test_socket(addr, TIMEOUT, false);

        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Failed),
        );
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_fail() {
        // We want to use here an address that will cause the connection to fail instantly.
        // On Windows and macOS we use an IPv6 black hole address,
        // as connections to 127.0.0.42:226 do not fail instantly.
        // On Linux we try to use a loopback address with an unused port,
        // because some hosts try connecting to the black hole address.
        let addr = if cfg!(target_os = "linux") {
            SocketAddr::from_str("127.0.0.42:226")
        } else {
            SocketAddr::from_str("[100::]:42")
        }
        .expect("test address");
        let (_client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);

        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Failed),
        );
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_server_close() {
        let (accept_task, addr) = start_test_server().await;
        let (_client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);

        let server_socket = accept_task.await.unwrap();
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Connected),
        );

        drop(server_socket);

        assert_next_socket_actions!(
            client_read;
            Close(dummy_handle!()),
        );
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_client_close() {
        let (accept_task, addr) = start_test_server().await;
        let (client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);

        let mut server_socket = accept_task.await.unwrap();
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Connected),
        );

        drop(client_write);

        assert_eq!(read_server(&mut server_socket).await, "");
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_basic_communication() {
        let (accept_task, addr) = start_test_server().await;
        let (client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);

        let mut server_socket = accept_task.await.unwrap();
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Connected),
        );

        write_server(&mut server_socket, "Hello World!").await;

        assert_next_socket_actions!(
            client_read;
            Data(dummy_handle!(), "Hello World!".as_bytes().to_vec()),
        );

        write_client(&client_write, "Hello from client").await;

        assert_eq!(read_server(&mut server_socket).await, "Hello from client");

        write_server(&mut server_socket, "from server 2").await;
        write_client(&client_write, "from client 2").await;

        assert_next_socket_actions!(
            client_read;
            Data(dummy_handle!(), "from server 2".as_bytes().to_vec()),
        );
        assert_eq!(read_server(&mut server_socket).await, "from client 2");
    }

    #[macro_rules_attribute::apply(async_test)]
    async fn test_socket_flush_before_close() {
        let (accept_task, addr) = start_test_server().await;
        let (client_write, client_read) = connect_test_socket(addr, TIMEOUT, true);

        let mut server_socket = accept_task.await.unwrap();
        assert_next_socket_actions!(
            client_read;
            Connect(dummy_handle!(), ConnectionState::Connected),
        );

        write_client(&client_write, "Sending some data").await;
        client_write.close();

        assert_eq!(read_server(&mut server_socket).await, "Sending some data");
    }

    /// AQW build: a server with one game file, as AQW's CDN serves them. Answers `304` to a
    /// request with the file's `ETag`, and counts the requests and the bodies it sent.
    async fn serve_game_file(
        listener: TcpListener,
        requests: Arc<Mutex<Vec<String>>>,
        bodies: Arc<std::sync::atomic::AtomicUsize>,
    ) {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut request = Vec::new();
            let mut buffer = [0; 1024];
            while !request.ends_with(b"\r\n\r\n") {
                match socket.read(&mut buffer).await {
                    Ok(n) if n > 0 => request.extend_from_slice(&buffer[..n]),
                    _ => break,
                }
            }
            let request = String::from_utf8_lossy(&request).to_lowercase();
            let headers =
                "cache-control: public, max-age=2678400\r\netag: \"v1\"\r\nconnection: close\r\n";
            let response = if request.contains("if-none-match: \"v1\"") {
                format!("HTTP/1.1 304 Not Modified\r\n{headers}\r\n")
            } else {
                bodies.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                format!("HTTP/1.1 200 OK\r\n{headers}content-length: 5\r\n\r\nhello")
            };
            requests
                .lock()
                .unwrap()
                .push(request.lines().next().unwrap_or_default().to_string());
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.shutdown().await;
        }
    }

    async fn fetch_chunks(
        backend: &ExternalNavigatorBackend<TestFutureSpawner, ()>,
        url: &str,
    ) -> Vec<u8> {
        let mut response = backend
            .fetch(Request::get(url.to_string()))
            .await
            .unwrap_or_else(|e| panic!("fetch {url}: {:?}", e.error));
        assert_eq!(response.status(), 200);
        assert_eq!(response.url(), url);
        let mut body = Vec::new();
        while let Some(chunk) = response.next_chunk().await.unwrap() {
            body.extend(chunk);
        }
        body
    }

    /// Waits for the cache's writes, which happen off the test's thread.
    async fn until(what: &str, done: impl Fn() -> bool) {
        for _ in 0..200 {
            if done() {
                return;
            }
            Timer::after(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for {what}");
    }

    async_test! {
        async fn http_cache_answers_and_revalidates() {
            use std::sync::atomic::{AtomicUsize, Ordering};

            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let port = listener.local_addr().unwrap().port();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let bodies = Arc::new(AtomicUsize::new(0));
            let server = task::spawn(serve_game_file(listener, requests.clone(), bodies.clone()));

            let dir = tempfile::tempdir().unwrap();
            let cache = HttpCache::new(dir.path(), 1 << 30).unwrap();
            let backend = new_test_backend(false).with_http_cache(Some(cache.clone()));
            let file = format!("http://127.0.0.1:{port}/gamefiles/mon/Frogzard.swf");
            let count = || requests.lock().unwrap().len();

            // The first request downloads the file, and it is kept.
            assert_eq!(fetch_chunks(&backend, &format!("{file}?v=5.32")).await, b"hello");
            until("the entry to be written", || cache.stats().stored == 1).await;
            assert_eq!((count(), bodies.load(Ordering::Relaxed)), (1, 1));

            // The same URL again: from disk, the server isn't asked. `body()` reads it too.
            assert_eq!(fetch_chunks(&backend, &format!("{file}?v=5.32")).await, b"hello");
            let response = backend
                .fetch(Request::get(format!("{file}?v=5.32")))
                .await
                .ok()
                .unwrap();
            assert_eq!(response.body().await.unwrap(), b"hello");
            assert_eq!(count(), 1);
            assert_eq!(cache.stats().hits, 2);

            // Another query string (the game after an update): the server is asked, says
            // "not modified", and sends no body.
            assert_eq!(fetch_chunks(&backend, &format!("{file}?v=5.33")).await, b"hello");
            assert_eq!((count(), bodies.load(Ordering::Relaxed)), (2, 1));
            assert_eq!(cache.stats().revalidated, 1);
            // From then on that query is answered from disk.
            let key = Url::parse(&format!("{file}?v=5.33")).unwrap();
            until("the entry to be refreshed", || {
                cache.lookup(&key).is_some_and(|entry| entry.meta.is_fresh(&key))
            })
            .await;
            assert_eq!(fetch_chunks(&backend, &format!("{file}?v=5.33")).await, b"hello");
            assert_eq!(count(), 2);

            // A POST, or a request with headers of its own, never touches the cache.
            let mut with_header = Request::get(format!("{file}?v=5.33"));
            with_header.set_headers(IndexMap::from([("x-test".to_string(), "1".to_string())]));
            let response = backend.fetch(with_header).await.ok().unwrap();
            assert_eq!(response.body().await.unwrap(), b"hello");
            assert_eq!((count(), bodies.load(Ordering::Relaxed)), (3, 2));

            // The server is gone: a URL that would be revalidated is answered from disk.
            server.abort();
            let _ = server.await;
            assert_eq!(fetch_chunks(&backend, &format!("{file}?v=5.34")).await, b"hello");
            // Without the cache the same request fails.
            let plain = new_test_backend(false);
            assert!(plain.fetch(Request::get(format!("{file}?v=5.34"))).await.is_err());
        }
    }
}
