use crate::backends::navigator::cache::CacheWriter;
use reqwest::Response as ReqwestResponse;
use ruffle_core::backend::navigator::{OwnedFuture, SuccessResponse};
use ruffle_core::loader::Error;
use ruffle_core::swf::Encoding;
use std::sync::{Arc, Mutex};

pub enum ResponseBody {
    /// The response's body comes from a file.
    File(Result<Vec<u8>, std::io::Error>),

    /// The response's body comes from the network.
    ///
    /// This has to be stored in shared ownership so that we can return
    /// owned futures. A synchronous lock is used here as we do not
    /// expect contention on this lock.
    Network(Arc<Mutex<Option<ReqwestResponse>>>),
}

pub struct Response {
    pub url: String,
    pub response_body: ResponseBody,
    pub text_encoding: Option<&'static Encoding>,
    pub status: u16,
    pub redirected: bool,
    /// AQW build: collects a network body for the disk cache as it is read (see `cache.rs`).
    pub cache: Option<Arc<Mutex<Option<CacheWriter>>>>,
}

/// AQW build: hands a chunk of a network body to the cache's writer, if there is one.
/// `None` is the end of the body, which stores it.
fn cache_chunk(cache: &Option<Arc<Mutex<Option<CacheWriter>>>>, chunk: Option<&[u8]>) {
    let Some(cache) = cache else {
        return;
    };
    let mut writer = cache.lock().expect("cache writer lock");
    match chunk {
        Some(chunk) => {
            if writer.as_mut().is_some_and(|writer| !writer.push(chunk)) {
                *writer = None;
            }
        }
        None => {
            if let Some(writer) = writer.take() {
                writer.finish();
            }
        }
    }
}

impl SuccessResponse for Response {
    fn url(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(&self.url)
    }

    fn set_url(&mut self, url: String) {
        self.url = url;
    }

    #[expect(clippy::await_holding_lock)]
    fn body(self: Box<Self>) -> OwnedFuture<Vec<u8>, Error> {
        match self.response_body {
            ResponseBody::File(file) => {
                Box::pin(async move { file.map_err(|e| Error::FetchError(e.to_string())) })
            }
            ResponseBody::Network(response) => {
                let cache = self.cache;
                Box::pin(async move {
                    let body = response
                        .lock()
                        .expect("working lock during fetch body read")
                        .take()
                        .expect("Body cannot already be consumed")
                        .bytes()
                        .await
                        .map_err(|e| Error::FetchError(e.to_string()))?
                        .to_vec();
                    cache_chunk(&cache, Some(&body));
                    cache_chunk(&cache, None);
                    Ok(body)
                })
            }
        }
    }

    fn text_encoding(&self) -> Option<&'static Encoding> {
        self.text_encoding
    }

    fn status(&self) -> u16 {
        self.status
    }

    fn redirected(&self) -> bool {
        self.redirected
    }

    #[expect(clippy::await_holding_lock)]
    fn next_chunk(&mut self) -> OwnedFuture<Option<Vec<u8>>, Error> {
        match &mut self.response_body {
            ResponseBody::File(file) => {
                let res = file
                    .as_mut()
                    .map(std::mem::take)
                    .map_err(|e| Error::FetchError(e.to_string()));

                Box::pin(async move {
                    match res {
                        Ok(bytes) if !bytes.is_empty() => Ok(Some(bytes)),
                        Ok(_) => Ok(None),
                        Err(e) => Err(e),
                    }
                })
            }
            ResponseBody::Network(response) => {
                let response = response.clone();
                let cache = self.cache.clone();
                Box::pin(async move {
                    let lock = response.try_lock();
                    if matches!(lock, Err(std::sync::TryLockError::WouldBlock)) {
                        return Err(Error::FetchError(
                            "Concurrent read operations on the same stream are not supported."
                                .to_string(),
                        ));
                    }

                    let result = lock
                        .expect("desktop network lock")
                        .as_mut()
                        .expect("Body cannot already be consumed")
                        .chunk()
                        .await;

                    match result {
                        Ok(Some(bytes)) => {
                            cache_chunk(&cache, Some(&bytes));
                            Ok(Some(bytes.to_vec()))
                        }
                        Ok(None) => {
                            cache_chunk(&cache, None);
                            Ok(None)
                        }
                        Err(e) => Err(Error::FetchError(e.to_string())),
                    }
                })
            }
        }
    }

    fn expected_length(&self) -> Result<Option<u64>, Error> {
        match &self.response_body {
            ResponseBody::File(file) => Ok(file.as_ref().map(|file| file.len() as u64).ok()),
            ResponseBody::Network(response) => {
                let lock = response.lock().expect("no recursive locks");
                let response = lock.as_ref().expect("Body cannot already be consumed");
                Ok(response.content_length())
            }
        }
    }
}
