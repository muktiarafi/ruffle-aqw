use crate::descriptors::Descriptors;
use crate::globals::Globals;
use fnv::FnvHashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

type PoolInner<T> = Mutex<Vec<T>>;
type Constructor<Type, Description> = Box<dyn Fn(&Descriptors, &Description) -> Type>;

#[derive(Debug)]
pub struct TexturePool {
    /// AQW build: each pool also remembers the frame it was last used in, see `end_frame`.
    pools: FnvHashMap<
        TextureKey,
        (
            BufferPool<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible>,
            u64,
        ),
    >,
    globals_cache: FnvHashMap<GlobalsKey, (Arc<Globals>, u64)>,
    frame: u64,
    /// AQW build: perf counters for textures taken / newly created, if counted.
    counters: Option<(usize, usize)>,
    /// AQW build: this pool's entry in `POOL_IDLE` (see `idle_budget`).
    id: u64,
    /// AQW build: whether the GPU shares the system's memory (integrated graphics), once
    /// known (see `idle_budget`).
    integrated: std::cell::Cell<Option<bool>>,
}

impl Default for TexturePool {
    fn default() -> Self {
        Self {
            pools: Default::default(),
            globals_cache: Default::default(),
            frame: 0,
            counters: None,
            id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
            integrated: std::cell::Cell::new(None),
        }
    }
}

impl Drop for TexturePool {
    fn drop(&mut self) {
        if let Ok(mut idle) = POOL_IDLE.lock() {
            idle.retain(|(id, _)| *id != self.id);
        }
    }
}

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// AQW build: every live pool's idle bytes, so the pools share one budget (see `idle_budget`).
static POOL_IDLE: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// AQW build: the idle textures all pools together may keep. Each pool had its own 512 MB
/// limit, and a boss fight filled both the layer pool and the cache pool at once: 1 GB of
/// textures nobody was using. On integrated graphics that is 1 GB of the system's RAM (the
/// GPU shares it), so the budget is halved there. `RUFFLE_POOL_IDLE_MB` overrides it (0 = no
/// limit), for comparisons; while the GPU's memory is nearly full, far less is kept.
fn idle_budget(integrated: bool) -> u64 {
    const DISCRETE_MB: u64 = 512;
    const INTEGRATED_MB: u64 = 256;
    const PRESSURE_MB: u64 = 64;
    static OVERRIDE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let limit = match OVERRIDE.get_or_init(|| {
        std::env::var("RUFFLE_POOL_IDLE_MB")
            .ok()
            .and_then(|mb| mb.parse().ok())
    }) {
        Some(mb) => *mb,
        None if integrated => INTEGRATED_MB,
        None => DISCRETE_MB,
    };
    LOGGED.get_or_init(|| {
        tracing::info!(
            target: "ruffle_aqw_perf",
            "idle texture budget: {limit} MB shared by the layer and cache pools (integrated graphics: {})",
            if integrated { "yes" } else { "no" }
        );
    });
    let limit = limit << 20;
    if MEMORY_PRESSURE.load(Ordering::Relaxed) {
        let pressure = PRESSURE_MB << 20;
        if limit == 0 { pressure } else { limit.min(pressure) }
    } else {
        limit
    }
}

impl TexturePool {
    pub fn new() -> Self {
        Default::default()
    }

    /// AQW build: a pool that counts its texture requests and creations in the perf log.
    pub fn new_counted(take_counter: usize, new_counter: usize) -> Self {
        let mut pool = Self::default();
        pool.counters = Some((take_counter, new_counter));
        pool
    }

    pub fn get_texture(
        &mut self,
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PoolEntry<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible> {
        let key = TextureKey {
            size,
            usage,
            format,
            sample_count,
        };
        if self.integrated.get().is_none() {
            self.integrated.set(Some(
                descriptors.adapter.get_info().device_type == wgpu::DeviceType::IntegratedGpu,
            ));
        }
        // AQW build: on a GPU that's out of memory (2 GB laptop cards in crowded AQW rooms) the
        // new texture is invalid, and using it was a fatal error. Free this pool's idle
        // textures, let the GPU release them, and try once more.
        let (entry, out_of_memory) =
            crate::catch_out_of_memory(&descriptors.device, || self.take_texture(descriptors, key));
        if out_of_memory {
            tracing::warn!(
                target: "ruffle_aqw_perf",
                "GPU out of memory for a {}x{} layer texture; freeing idle textures and retrying",
                size.width,
                size.height
            );
            // Don't let the invalid texture go back into a pool.
            self.pools.clear();
            drop(entry);
            crate::release_gpu_memory(descriptors);
            return self.take_texture(descriptors, key);
        }
        entry
    }

    fn take_texture(
        &mut self,
        descriptors: &Descriptors,
        key: TextureKey,
    ) -> PoolEntry<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible> {
        let TextureKey {
            size,
            usage,
            format,
            sample_count,
        } = key;
        let frame = self.frame;
        let (pool, last_used) = self.pools.entry(key).or_insert_with(|| {
            let label = if cfg!(feature = "render_debug_labels") {
                use std::sync::atomic::{AtomicU32, Ordering};
                static ID_COUNT: AtomicU32 = AtomicU32::new(0);
                let id = ID_COUNT.fetch_add(1, Ordering::Relaxed);
                create_debug_label!("Pooled texture {}", id)
            } else {
                None
            };
            (BufferPool::new(Box::new(move |descriptors, _description| {
                let _timer = crate::perf_counters::timers::Scoped::new(
                    crate::perf_counters::timers::TEXTURE_CREATE,
                );
                // AQW build: pooled textures come and go; they get GPU memory of their own.
                let texture = crate::texture_kind(crate::TRANSIENT_TEXTURES, || {
                    descriptors.device.create_texture(&wgpu::TextureDescriptor {
                        label: label.as_deref(),
                        size,
                        mip_level_count: 1,
                        sample_count,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        view_formats: &[format],
                        usage,
                    })
                });
                let view = texture.create_view(&Default::default());
                (texture, view)
            })), 0)
        });
        *last_used = frame;
        if let Some((take_counter, new_counter)) = self.counters {
            crate::perf_counters::inc(take_counter);
            if pool.available.lock().is_ok_and(|v| v.is_empty()) {
                crate::perf_counters::inc(new_counter);
            }
        }
        pool.take(descriptors, AlwaysCompatible)
    }

    pub fn get_globals(
        &mut self,
        descriptors: &Descriptors,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Arc<Globals> {
        let key = GlobalsKey {
            viewport_width,
            viewport_height,
        };
        if let Some((globals, last_used)) = self.globals_cache.get_mut(&key) {
            *last_used = self.frame;
            return globals.clone();
        }
        // AQW build: if the GPU is out of memory for the buffer (phones, whose GPU shares the
        // RAM), frees this pool's idle textures, lets the GPU release them and tries once more.
        // One that still failed is used for this frame (it draws nothing) but not cached, so the
        // next frame tries again.
        let create = || {
            crate::catch_out_of_memory(&descriptors.device, || {
                Globals::new(
                    &descriptors.device,
                    &descriptors.bind_layouts.globals,
                    viewport_width,
                    viewport_height,
                )
            })
        };
        let (mut globals, mut out_of_memory) = create();
        if out_of_memory {
            tracing::warn!(
                target: "ruffle_aqw_perf",
                "GPU out of memory for the globals of a {viewport_width}x{viewport_height} target; freeing idle textures and retrying"
            );
            drop(globals);
            self.clear_idle();
            crate::release_gpu_memory(descriptors);
            (globals, out_of_memory) = create();
        }
        let globals = Arc::new(globals);
        if out_of_memory {
            tracing::warn!(
                target: "ruffle_aqw_perf",
                "GPU still out of memory for the globals of a {viewport_width}x{viewport_height} target"
            );
        } else {
            self.globals_cache
                .insert(key, (globals.clone(), self.frame));
        }
        globals
    }

    /// AQW build: frees the textures (and cached globals) of sizes that weren't used in the
    /// last `max_idle_frames` frames. Pools are keyed by exact size, so without this every
    /// distinct filter/blend layer size ever drawn keeps its textures for the whole session
    /// (gigabytes of VRAM after a while in AQW, plus more resources for wgpu to track).
    /// Textures still in use are unaffected: they're dropped instead of returned to the pool.
    /// AQW build, for the perf log: (sizes, idle pooled textures, their approximate bytes).
    pub fn stats(&self) -> (usize, usize, u64) {
        let mut textures = 0;
        let mut bytes = 0;
        for (key, (pool, _)) in &self.pools {
            textures += pool.available.lock().map(|v| v.len()).unwrap_or(0);
            bytes += idle_bytes(key, pool);
        }
        (self.pools.len(), textures, bytes)
    }

    /// AQW build: frees every idle pooled texture (textures in use are dropped when released).
    pub fn clear_idle(&mut self) {
        self.pools.clear();
    }

    pub fn end_frame(&mut self, max_idle_frames: u64) {
        let oldest = self.frame.saturating_sub(max_idle_frames);
        self.pools.retain(|_, (_, last_used)| *last_used >= oldest);
        self.globals_cache
            .retain(|_, (_, last_used)| *last_used >= oldest);
        self.limit_idle_bytes();
        self.frame += 1;
    }

    /// AQW build: some AQW effects (e.g. boss fights in ultradrakath) draw ~1800 temporary
    /// textures a second, nearly all of different sizes, so ~1 s of idle textures was up to
    /// 5 GB of VRAM that is almost never reused (filling 4 GB cards). Keeps the idle textures
    /// of all pools together within `idle_budget` by freeing this pool's least recently used
    /// sizes first, never the ones the last two frames used, so steady reuse is unaffected.
    fn limit_idle_bytes(&mut self) {
        let budget = idle_budget(self.integrated.get().unwrap_or(false));
        let mut sizes: Vec<(u64, TextureKey, u64)> = self
            .pools
            .iter()
            .map(|(key, (pool, last_used))| (*last_used, *key, idle_bytes(key, pool)))
            .collect();
        let mut total: u64 = sizes.iter().map(|(_, _, bytes)| bytes).sum();
        if budget != 0 {
            // What the other pools keep idle comes out of this pool's share.
            let others: u64 = POOL_IDLE
                .lock()
                .map(|idle| {
                    idle.iter()
                        .filter(|(id, _)| *id != self.id)
                        .map(|(_, bytes)| bytes)
                        .sum()
                })
                .unwrap_or(0);
            let allowed = budget.saturating_sub(others);
            if total > allowed {
                sizes.sort_unstable_by_key(|(last_used, _, _)| *last_used);
                // Sizes the last two frames used are kept whatever the limit: a crowded
                // scene that really reuses more than the limit every frame keeps all of it
                // (no recreating textures every frame); only leftovers that aren't being
                // reused are freed.
                let keep_from = self.frame.saturating_sub(1);
                for (last_used, key, bytes) in sizes {
                    if total <= allowed || last_used >= keep_from {
                        break;
                    }
                    if bytes > 0 {
                        self.pools.remove(&key);
                        total -= bytes;
                    }
                }
            }
        }
        if let Ok(mut idle) = POOL_IDLE.lock() {
            match idle.iter_mut().find(|(id, _)| *id == self.id) {
                Some(entry) => entry.1 = total,
                None => idle.push((self.id, total)),
            }
        }
    }
}

/// AQW build: set while the GPU's memory is nearly full (see `set_memory_pressure`).
pub(crate) static MEMORY_PRESSURE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Approximate bytes of a pool's idle textures.
fn idle_bytes(
    key: &TextureKey,
    pool: &BufferPool<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible>,
) -> u64 {
    let idle = pool.available.lock().map(|v| v.len()).unwrap_or(0);
    idle as u64
        * key.size.width as u64
        * key.size.height as u64
        * key.format.block_copy_size(None).unwrap_or(4) as u64
        * key.sample_count as u64
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct TextureKey {
    size: wgpu::Extent3d,
    usage: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
    sample_count: u32,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct GlobalsKey {
    viewport_width: u32,
    viewport_height: u32,
}

pub trait BufferDescription: Clone + Debug {
    type Cost: Ord;

    /// If the potential buffer represented by this description (`self`)
    /// fits another existing buffer and its description (`other`),
    /// return the cost to use that buffer instead of making a new one.
    ///
    /// Cost is an arbitrary unit, but lower is better.
    /// None means that the other buffer cannot be used in place of this one.
    fn cost_to_use(&self, other: &Self) -> Option<Self::Cost>;
}

#[derive(Clone, Debug)]
pub struct AlwaysCompatible;

impl BufferDescription for AlwaysCompatible {
    type Cost = ();

    fn cost_to_use(&self, _other: &Self) -> Option<()> {
        Some(())
    }
}

pub struct BufferPool<Type, Description: BufferDescription> {
    available: Arc<PoolInner<(Type, Description)>>,
    constructor: Constructor<Type, Description>,
}

impl<Type, Description: BufferDescription> Debug for BufferPool<Type, Description> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool").finish()
    }
}

impl<Type, Description: BufferDescription> BufferPool<Type, Description> {
    pub fn new(constructor: Constructor<Type, Description>) -> Self {
        Self {
            available: Arc::new(Mutex::new(vec![])),
            constructor,
        }
    }

    pub fn take(
        &self,
        descriptors: &Descriptors,
        description: Description,
    ) -> PoolEntry<Type, Description> {
        let mut guard = self
            .available
            .lock()
            .expect("Should not be able to lock recursively");
        let mut best: Option<(Description::Cost, usize)> = None;
        for i in 0..guard.len() {
            if let Some(cost) = description.cost_to_use(&guard[i].1) {
                if let Some(best) = &mut best {
                    if best.0 > cost {
                        *best = (cost, i);
                    }
                } else if best.is_none() {
                    best = Some((cost, i));
                }
            }
        }

        let (item, used_description) = if let Some((_, best)) = best {
            guard.swap_remove(best)
        } else {
            let item = (self.constructor)(descriptors, &description);
            (item, description)
        };
        PoolEntry {
            item: Some(item),
            description: used_description,
            pool: Arc::downgrade(&self.available),
        }
    }
}

pub struct PoolEntry<Type, Description: BufferDescription> {
    item: Option<Type>,
    description: Description,
    pool: Weak<PoolInner<(Type, Description)>>,
}

impl<Type, Description: BufferDescription> Debug for PoolEntry<Type, Description>
where
    Type: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PoolEntry").field(&self.item).finish()
    }
}

impl<Type, Description: BufferDescription> Drop for PoolEntry<Type, Description> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take()
            && let Some(pool) = self.pool.upgrade()
        {
            pool.lock()
                .expect("Should not be able to lock recursively")
                .push((item, self.description.clone()))
        }
    }
}

impl<Type, Description: BufferDescription> Deref for PoolEntry<Type, Description> {
    type Target = Type;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("Item should exist until dropped")
    }
}
