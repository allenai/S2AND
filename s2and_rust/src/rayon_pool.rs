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

// mimalloc keeps a block that one thread frees on another thread's page until
// the owning thread collects. Keep this cfg equal to the global allocator's
// cfg in lib.rs.
#[cfg(not(any(target_env = "musl", all(target_os = "linux", target_arch = "aarch64"))))]
extern "C" {
    fn mi_collect(force: bool);
}

/// Return the memory freed on every cached pool worker's pages to the OS.
///
/// Background release frees a request's rows from a thread that did not
/// allocate them. mimalloc parks such frees on the owning worker's pages, and
/// idle workers never collect them, so the process would keep the request's
/// memory until its next request. A forced collect on each worker takes the
/// parked frees and returns empty pages at once.
pub(crate) fn collect_freed_memory_on_pool_workers() {
    #[cfg(not(any(target_env = "musl", all(target_os = "linux", target_arch = "aarch64"))))]
    {
        let pools: Vec<Arc<rayon::ThreadPool>> = match rayon_pool_cache().lock() {
            Ok(cache) => cache
                .pools
                .values()
                .map(|entry| Arc::clone(&entry.pool))
                .collect(),
            // A poisoned cache only skips the collection; memory returns on the next request.
            Err(_) => return,
        };
        for pool in pools {
            // SAFETY: mi_collect touches only the calling thread's heap and the shared arenas.
            pool.broadcast(|_| unsafe { mi_collect(true) });
        }
    }
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

#[cfg(test)]
mod tests {
    use super::{cached_rayon_pool, collect_freed_memory_on_pool_workers};

    #[test]
    fn collecting_after_cross_thread_frees_keeps_pools_usable() {
        let pool = cached_rayon_pool(3).expect("pool of three threads");
        // Allocate on the workers, then free from this thread, as background release does.
        let blocks: Vec<Vec<u64>> = pool.install(|| {
            use rayon::prelude::*;
            (0..64)
                .into_par_iter()
                .map(|index| vec![index; 4096])
                .collect()
        });
        drop(blocks);
        collect_freed_memory_on_pool_workers();
        let total: u64 = pool.install(|| {
            use rayon::prelude::*;
            (0..64u64).into_par_iter().map(|index| index * 2).sum()
        });
        assert_eq!(total, 4032);
    }
}
