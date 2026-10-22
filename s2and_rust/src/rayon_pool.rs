use crate::HashMap;
use rayon::ThreadPoolBuilder;
use std::sync::{Arc, Mutex, OnceLock};

const RAYON_POOL_CACHE_MAX_ENTRIES: usize = 8;

struct CachedRayonPool {
    pool: Arc<rayon::ThreadPool>,
    last_used: u64,
}

struct RayonPoolCache {
    pools: HashMap<usize, CachedRayonPool>,
    access_counter: u64,
}

impl RayonPoolCache {
    fn new() -> Self {
        Self {
            pools: HashMap::default(),
            access_counter: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.access_counter = self.access_counter.wrapping_add(1);
        self.access_counter
    }
}

static RAYON_POOL_CACHE: OnceLock<Mutex<RayonPoolCache>> = OnceLock::new();

fn rayon_pool_cache() -> &'static Mutex<RayonPoolCache> {
    RAYON_POOL_CACHE.get_or_init(|| Mutex::new(RayonPoolCache::new()))
}

fn cached_rayon_pool(thread_count: usize) -> Option<Arc<rayon::ThreadPool>> {
    if thread_count == 0 {
        return None;
    }
    if let Ok(mut cache) = rayon_pool_cache().lock() {
        let touch = cache.tick();
        if let Some(entry) = cache.pools.get_mut(&thread_count) {
            entry.last_used = touch;
            return Some(Arc::clone(&entry.pool));
        }
    }

    let built_pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .ok()?;
    let built_pool = Arc::new(built_pool);
    if let Ok(mut cache) = rayon_pool_cache().lock() {
        if cache.pools.len() >= RAYON_POOL_CACHE_MAX_ENTRIES
            && !cache.pools.contains_key(&thread_count)
        {
            if let Some(victim_key) = cache
                .pools
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
            {
                cache.pools.remove(&victim_key);
            }
        }
        let touch = cache.tick();
        let pooled = cache
            .pools
            .entry(thread_count)
            .or_insert_with(|| CachedRayonPool {
                pool: Arc::clone(&built_pool),
                last_used: touch,
            });
        pooled.last_used = touch;
        return Some(Arc::clone(&pooled.pool));
    }
    Some(built_pool)
}

pub(crate) fn install_with_optional_rayon_pool<T, F>(num_threads: Option<usize>, compute: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    if let Some(thread_count) = num_threads {
        let threads = thread_count.max(1);
        if let Some(pool) = cached_rayon_pool(threads) {
            return pool.install(compute);
        }
    }
    compute()
}
