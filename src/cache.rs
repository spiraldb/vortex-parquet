// SPDX-License-Identifier: Apache-2.0

//! In-flight decode sharing with opt-in retained caching.
//!
//! # Why this is not per-leaf
//!
//! The default stores weak references. A scan lowers a field once per filter conjunct and once for
//! its projection, and every one of those plans takes the same entry when it is lowered, so they
//! share one decode; concurrent scans of the same row group share it too. Once the plans release
//! their `Arc`s, the array is immediately reclaimable instead of remaining resident for the scan. Explicit bounded and
//! unbounded policies retain completed arrays for callers that knowingly want a data cache.
//!
//! # What the budget is measured in
//!
//! Retained policies count **compressed** bytes: the column chunk's on-disk length. That is knowable
//! before the decode, whereas the decoded size is not — the entry would have to be admitted before it
//! could be measured, which is the one thing a budget cannot allow. Decoded size is a roughly constant
//! multiple of compressed size within a column, so this orders entries correctly even though the
//! absolute number is not the memory actually held. Treat the budget as a dial to turn, not as an
//! allocator limit; if it needs to become a true byte budget later, that wants the decoded size
//! reported back on completion.

use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex::utils::aliases::hash_map::HashMap;

/// How much decoded data to keep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CachePolicy {
    /// Share decodes only while a consumer or an in-flight task still holds them.
    ///
    /// This is the production default. It coalesces concurrent filter and projection requests but
    /// does not turn a scan into an implicit decoded-data cache: once the last consumer drops its
    /// `Arc`, the decoded array is immediately reclaimable.
    #[default]
    Ephemeral,
    /// Keep nothing at all. Every evaluation decodes the chunk again.
    ///
    /// Unlike [`Self::Ephemeral`], this does not coalesce concurrent evaluations. Use it to measure
    /// execution without decode sharing.
    Off,
    /// Keep up to this many compressed-equivalent bytes, evicting least-recently-used first.
    Bounded(u64),
    /// Keep everything. The gas pedal.
    ///
    /// For a wide scan of a large file this is "hold the decoded file in memory", which is the intent
    /// rather than an accident — but it is why it is not the default.
    JustSendIt,
}

/// Identifies the file a cached chunk came from.
///
/// Assigned once per open, so independent opens have distinct cache identities even when their paths
/// match. Cache entries from one opened file cannot supply another opened file's decoded arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileId(u64);

impl FileId {
    /// Allocates an identity no other open will be given.
    pub fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// What a cache entry holds the decode *of*.
///
/// # Why this is not a hash
///
/// A leaf covering one column decodes one column; a leaf covering a row group decodes a *set* of them,
/// chosen per expression. Both need to key the cache, so the key has to carry either shape.
///
/// The first version of this hashed the column set into a `usize`. That is wrong, and not merely
/// theoretically: a hash is not injective, so two different projections can collide and the cache will
/// serve one projection's decode for the other's — the right number of rows, the wrong columns, no
/// error. Sample assertions that two particular sets hash differently cannot establish otherwise.
///
/// So the exact set is carried. It is an `Arc<[usize]>` because the key is cloned into the map and a
/// wide projection is a hundred-odd indices; comparing them on a hit is cheaper than a decode by
/// orders of magnitude.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DecodeUnit {
    /// One top-level column, by index.
    Column(usize),
    /// A set of top-level columns, **sorted and deduplicated** by the caller so that two requests for
    /// the same columns compare equal whatever order they arrived in.
    Columns(Arc<[usize]>),
}

/// Identifies a cached decode: which file, which row group, and what was decoded.
///
/// The file identity is load-bearing. `ParquetFormat` owns the cache and every file it opens shares it,
/// so a key of only (row group, column) lets one file's chunk be served from another's — silently wrong
/// values when the shapes happen to match. A single-table benchmark cannot catch that, which is exactly
/// how it survived review here once already.
pub type ChunkKey = (FileId, usize, DecodeUnit);

/// A cache of in-flight or completed decodes, keyed by (row group, top-level column).
pub struct DecodeCache<T> {
    policy: CachePolicy,
    inner: Mutex<Inner<T>>,
}

struct Inner<T> {
    entries: HashMap<ChunkKey, Entry<T>>,
    /// Ticks once per access, so `Entry::last_used` orders entries without maintaining a list.
    ///
    /// A `Vec` in LRU order was the obvious implementation and it made a cache hit a linear search
    /// followed by `Vec::remove`. Under `JustSendIt` over a file with tens of thousands of chunks, a warm
    /// scan touching them in order is then quadratic.
    clock: u64,
    charged: u64,
    accesses_since_sweep: usize,
}

struct Entry<T> {
    value: CacheValue<T>,
    /// The chunk's compressed length, which is what the budget counts.
    charge: u64,
    /// Clock value at the last access.
    last_used: u64,
}

enum CacheValue<T> {
    Ephemeral(Weak<T>),
    Retained(Arc<T>),
}

impl<T> CacheValue<T> {
    fn upgrade(&self) -> Option<Arc<T>> {
        match self {
            Self::Ephemeral(value) => value.upgrade(),
            Self::Retained(value) => Some(Arc::clone(value)),
        }
    }

    fn is_live(&self) -> bool {
        match self {
            Self::Ephemeral(value) => value.strong_count() != 0,
            Self::Retained(_) => true,
        }
    }
}

impl<T> std::fmt::Debug for DecodeCache<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock();
        f.debug_struct("DecodeCache")
            .field("policy", &self.policy)
            .field("entries", &inner.entries.len())
            .field("charged", &inner.charged)
            .finish()
    }
}

impl<T> DecodeCache<T> {
    /// Creates a cache with the given policy.
    pub fn new(policy: CachePolicy) -> Self {
        Self {
            policy,
            inner: Mutex::new(Inner {
                entries: HashMap::default(),
                clock: 0,
                charged: 0,
                accesses_since_sweep: 0,
            }),
        }
    }

    /// The policy this cache was built with.
    pub fn policy(&self) -> CachePolicy {
        self.policy
    }

    /// Drops everything held.
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.entries.clear();
        inner.charged = 0;
        inner.accesses_since_sweep = 0;
    }

    /// Whether a live decode for `key` is already held, without creating one.
    pub fn contains(&self, key: ChunkKey) -> bool {
        if self.policy == CachePolicy::Off {
            return false;
        }
        self.inner
            .lock()
            .entries
            .get(&key)
            .and_then(|entry| entry.value.upgrade())
            .is_some()
    }

    /// Returns a currently live decode without creating one.
    pub fn get(&self, key: &ChunkKey) -> Option<Arc<T>> {
        if self.policy == CachePolicy::Off {
            return None;
        }
        let mut inner = self.inner.lock();
        inner.clock += 1;
        let now = inner.clock;
        let entry = inner.entries.get_mut(key)?;
        let value = entry.value.upgrade()?;
        entry.last_used = now;
        Some(value)
    }

    /// Returns the decode for `key`, creating it with `decode` if it is not already held.
    ///
    /// `charge` is the chunk's compressed length. With [`CachePolicy::Off`] nothing is retained and
    /// `decode` runs on every call — see that variant's note on how it differs from the `Weak` this
    /// replaced.
    pub fn get_or_insert(&self, key: ChunkKey, charge: u64, decode: impl FnOnce() -> T) -> Arc<T> {
        if self.policy == CachePolicy::Off {
            return Arc::new(decode());
        }

        let mut inner = self.inner.lock();
        inner.clock += 1;
        let now = inner.clock;
        if let Some(entry) = inner.entries.get_mut(&key)
            && let Some(value) = entry.value.upgrade()
        {
            entry.last_used = now;
            return value;
        }

        if let Some(stale) = inner.entries.remove(&key) {
            inner.charged = inner.charged.saturating_sub(stale.charge);
        }

        // Held across the decode *construction*, not the decode itself: `decode` builds a future, it
        // does not await it. Two evaluations of one chunk therefore share one future rather than racing.
        let value = Arc::new(decode());
        let (cached, charge) = match self.policy {
            CachePolicy::Ephemeral => (CacheValue::Ephemeral(Arc::downgrade(&value)), 0),
            CachePolicy::Bounded(_) | CachePolicy::JustSendIt => {
                (CacheValue::Retained(Arc::clone(&value)), charge)
            }
            CachePolicy::Off => unreachable!("off returned before locking"),
        };
        inner.entries.insert(
            key.clone(),
            Entry {
                value: cached,
                charge,
                last_used: now,
            },
        );
        inner.charged = inner.charged.saturating_add(charge);

        inner.accesses_since_sweep += 1;
        if self.policy == CachePolicy::Ephemeral && inner.accesses_since_sweep >= 256 {
            inner.entries.retain(|_, entry| entry.value.is_live());
            inner.accesses_since_sweep = 0;
        }

        if let CachePolicy::Bounded(budget) = self.policy {
            inner.evict_to(budget, &key);
        }

        value
    }
}

impl<T> Inner<T> {
    /// Drops least-recently-used entries until the charge fits `budget`.
    ///
    /// `keep` is never evicted: it was just inserted and the caller is about to use it, so evicting it
    /// would decode it again immediately. A budget smaller than a single chunk therefore holds one
    /// chunk rather than thrashing, which is the useful reading of a too-small budget.
    ///
    /// Scanning for the oldest entry each time is O(n) per eviction, but eviction only happens when the
    /// budget is exceeded, whereas a hit is now O(1) — which is the ratio that matters.
    fn evict_to(&mut self, budget: u64, keep: &ChunkKey) {
        while self.charged > budget {
            let oldest = self
                .entries
                .iter()
                .filter(|(key, _)| *key != keep)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone());
            let Some(key) = oldest else {
                return;
            };
            if let Some(entry) = self.entries.remove(&key) {
                self.charged = self.charged.saturating_sub(entry.charge);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One file identity, reused across a test so keys collide as they would in one file.
    fn f() -> FileId {
        static ONE: std::sync::OnceLock<FileId> = std::sync::OnceLock::new();
        *ONE.get_or_init(FileId::next)
    }

    /// A column-set key must compare by its contents, exactly.
    ///
    /// The point of `DecodeUnit::Columns` carrying the set rather than a hash of it: distinct sets are
    /// distinct keys with no collision to reason about, and equal sets are equal keys.
    #[test]
    fn a_column_set_key_compares_by_contents() {
        let wide = DecodeUnit::Columns(Arc::from(vec![0usize, 3, 7]));
        let same = DecodeUnit::Columns(Arc::from(vec![0usize, 3, 7]));
        let narrow = DecodeUnit::Columns(Arc::from(vec![0usize, 3]));
        assert_eq!(wide, same);
        assert_ne!(wide, narrow);
        // And a single column is never confusable with the set containing only it.
        assert_ne!(
            DecodeUnit::Column(3),
            DecodeUnit::Columns(Arc::from(vec![3usize]))
        );
    }

    /// Two different column sets must not share an entry.
    #[test]
    fn different_column_sets_do_not_share_an_entry() {
        let cache: DecodeCache<u32> = DecodeCache::new(CachePolicy::JustSendIt);
        let wide = DecodeUnit::Columns(Arc::from(vec![0usize, 1]));
        let narrow = DecodeUnit::Columns(Arc::from(vec![0usize]));

        assert_eq!(*cache.get_or_insert((f(), 0, wide.clone()), 10, || 1), 1);
        assert_eq!(*cache.get_or_insert((f(), 0, narrow.clone()), 10, || 2), 2);
        assert_eq!(*cache.get_or_insert((f(), 0, wide), 10, || 99), 1);
        assert_eq!(*cache.get_or_insert((f(), 0, narrow), 10, || 99), 2);
    }

    /// The same key must decode once, which is the whole purpose.
    #[test]
    fn a_repeated_key_decodes_once() {
        let cache: DecodeCache<u32> = DecodeCache::new(CachePolicy::Bounded(1000));
        let mut decodes = 0;
        for _ in 0..5 {
            let value = cache.get_or_insert((f(), 0, DecodeUnit::Column(0)), 10, || {
                decodes += 1;
                7
            });
            assert_eq!(*value, 7);
        }
        assert_eq!(decodes, 1, "a cached chunk must not be decoded again");
    }

    /// `Off` must reproduce the pre-cache behaviour exactly: nothing retained.
    #[test]
    fn the_off_policy_retains_nothing() {
        let cache: DecodeCache<u32> = DecodeCache::new(CachePolicy::Off);
        let mut decodes = 0;
        for _ in 0..3 {
            cache.get_or_insert((f(), 0, DecodeUnit::Column(0)), 10, || {
                decodes += 1;
                7
            });
        }
        assert_eq!(decodes, 3);
        assert!(!cache.contains((f(), 0, DecodeUnit::Column(0))));
    }

    /// Ephemeral sharing must not retain decoded data after the active consumers release it.
    #[test]
    fn ephemeral_cache_releases_completed_decodes() {
        let cache: DecodeCache<u32> = DecodeCache::new(CachePolicy::Ephemeral);
        let key = (f(), 0, DecodeUnit::Column(0));
        let value = cache.get_or_insert(key.clone(), 10, || 7);
        let weak = Arc::downgrade(&value);
        assert!(cache.contains(key.clone()));
        drop(value);
        assert!(weak.upgrade().is_none());
        assert!(!cache.contains(key));
    }

    /// Eviction must be least-recently-used, and must respect the budget.
    #[test]
    fn eviction_drops_the_least_recently_used() {
        let cache: DecodeCache<u32> = DecodeCache::new(CachePolicy::Bounded(25));
        cache.get_or_insert((f(), 0, DecodeUnit::Column(0)), 10, || 1);
        cache.get_or_insert((f(), 0, DecodeUnit::Column(1)), 10, || 2);
        // Touch the first so the second becomes least-recently-used.
        cache.get_or_insert((f(), 0, DecodeUnit::Column(0)), 10, || 1);
        cache.get_or_insert((f(), 0, DecodeUnit::Column(2)), 10, || 3);

        assert!(
            cache.contains((f(), 0, DecodeUnit::Column(0))),
            "recently used entry was evicted"
        );
        assert!(
            cache.contains((f(), 0, DecodeUnit::Column(2))),
            "the just-inserted entry was evicted"
        );
        assert!(
            !cache.contains((f(), 0, DecodeUnit::Column(1))),
            "least-recently-used entry survived"
        );
    }

    /// A budget below one chunk holds that chunk rather than evicting what the caller is about to use.
    #[test]
    fn a_budget_smaller_than_one_chunk_still_holds_it() {
        let cache: DecodeCache<u32> = DecodeCache::new(CachePolicy::Bounded(1));
        cache.get_or_insert((f(), 0, DecodeUnit::Column(0)), 1000, || 1);
        assert!(cache.contains((f(), 0, DecodeUnit::Column(0))));
    }

    /// The gas pedal keeps everything.
    #[test]
    fn just_send_it_evicts_nothing() {
        let cache: DecodeCache<u32> = DecodeCache::new(CachePolicy::JustSendIt);
        for col in 0..50 {
            cache.get_or_insert((f(), 0, DecodeUnit::Column(col)), 1_000_000, || {
                u32::try_from(col).unwrap()
            });
        }
        assert!((0..50).all(|col| cache.contains((f(), 0, DecodeUnit::Column(col)))));
    }
}
