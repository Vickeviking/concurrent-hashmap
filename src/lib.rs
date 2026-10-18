mod node;
use node::*;

use crossbeam::epoch::{Atomic, Guard, Owned, Shared};
use std::hash::{BuildHasher, Hash, Hasher, RandomState};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

use crate::node::BinEntry;

/// The largest possible table capacity.  This value must be
/// exactly 1<<30 to stay within Java array allocation and indexing
/// bounds for power of two table sizes, and is further required
/// because the top two bits of 32bit hash fields are used for
/// control purposes.
const MAXIMUM_CAPACITY: usize = 1 << 30;

/// The default initial table capacity.  Must be a power of 2
/// (i.e., at least 1) and at most MAXIMUM_CAPACITY.
const DEFAULT_CAPACITY: usize = 16;

/// The load factor for this table. Overrides of this value in
/// constructors affect only the initial table capacity.  The
/// actual floating point value isn't normally used -- it is
/// simpler to use expressions such as {@code n - (n >>> 2)} for
/// the associated resizing threshold.
const LOAD_FACTOR: f64 = 0.75;

/// Minimum number of rebinnings per transfer step. Ranges are
/// subdivided to allow multiple resizer threads.  This value
/// serves as a lower bound to avoid resizers encountering
/// excessive memory contention.  The value should be at least
/// `DEFAULT_CAPACITY`.
const MIN_TRANSFER_STRIDE: isize = 16;

/// The number of bits used for generation stamp in `sizeCtl`.
/// Must be at least 6 for 32bit arrays.
const RESIZE_STAMP_BITS: usize = 16;

/// The maximum number of threads that can help resize.
/// Must fit in 32 - `RESIZE_STAMP_BITS bits`.
const MAX_RESIZERS: isize = (1 << (32 - RESIZE_STAMP_BITS)) - 1;

/// The bit shift for recording size stamp in `sizeCtl`.
const RESIZE_STAMP_SHIFT: usize = 32 - RESIZE_STAMP_BITS;

pub struct ConcurrentHashMap<K, V, S = RandomState> {
    /// The array of bins. Lazily initialized upon first insertion.
    /// Size is always a power of two. Accessed directly by iterators.
    table: Atomic<Table<K, V>>,

    /// The next table to use; non-null only while resizing.
    next_table: Atomic<Table<K, V>>,

    /// The next table index (Plus one) to split while resizing.
    transfer_index: AtomicIsize,

    count: AtomicUsize,

    /// Table initialization and resizing control.  When negative, the
    /// table is being initialized or resized: -1 for initialization,
    /// else -(1 + the number of active resizing threads).  Otherwise,
    /// when table is null, holds the initial table size to use upon
    /// creation, or 0 for default. After initialization, holds the
    /// next element count value upon which to resize the table.
    size_ctl: AtomicIsize,

    build_hasher: S, // A thing that builds hashers
}

impl<K, V, S> ConcurrentHashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    fn hash(&self, key: &K) -> u64 {
        let mut h = self.build_hasher.build_hasher();
        key.hash(&mut h);
        h.finish()
    }

    /// Returns `None` if the key is absent; never returns `Some(Shared::null())`.
    pub fn get<'g>(&'g self, key: &K, guard: &'g Guard) -> Option<Shared<'g, V>> {
        let h = self.hash(key);
        let table = self.table.load(Ordering::SeqCst, guard);
        if table.is_null() {
            return None;
        }
        // SAFETY: non-null, and tables are only reclaimed through epoch-deferred
        // destruction, so it stays valid while `guard` is pinned.
        let table = unsafe { table.deref() };
        if table.bins.len() == 0 {
            return None;
        }

        let bini = table.bin_index(h);
        let bin = table.bin(bini, guard);
        if bin.is_null() {
            return None;
        }
        // SAFETY: non-null, and loaded under `guard` (same argument as `table`).
        let bin = unsafe { bin.deref() };

        let node = bin.find(h, key, guard);
        if node.is_null() {
            return None;
        }
        // SAFETY: non-null, and loaded under `guard` (same argument as `table`).
        let node = unsafe { node.deref() };

        let v = node.value.load(Ordering::SeqCst, guard);
        //INFO: Why do this? can it happen?
        assert!(!v.is_null());
        Some(v)
    }

    pub fn get_and<R, F: FnOnce(&V) -> R>(&self, key: &K, then: F) -> Option<R> {
        let guard = &crossbeam::epoch::pin();
        self.get(key, guard).map(|v| {
            // SAFETY: `get` only returns `Some` for a non-null pointer to a value that
            // was reachable from the map when it was loaded under `guard`. A value is
            // only reclaimed through epoch-deferred destruction, and `guard` stays
            // pinned until this function returns, so the value outlives the call to
            // `then`. `then` cannot keep the `&V` past that call, since `R` cannot
            // borrow from the argument.
            then(unsafe { v.deref() })
        })
    }

    // You can only claim V as your own, once everyone else has dropped it
    pub fn insert(&self, key: K, value: V) -> Option<()> {
        // allow replace
        self.put(key, value, false)
    }

    fn put(&self, key: K, value: V, no_replacement: bool) -> Option<()> {
        let h = self.hash(&key);

        let guard = &crossbeam::epoch::pin();
        let mut table = self.table.load(Ordering::SeqCst, guard);

        let mut node = Owned::new(BinEntry::Node(Node {
            key,
            value: Owned::new(value),
            hash: h,
            next: Atomic::null(),
        }));

        // Let node own key, but then take a reference to it for later comparison
        let key = if let BinEntry::Node(Node { ref key, .. }) = *node {
            key
        } else {
            unreachable!();
        };

        loop {
            // SAFETY: only dereferenced when non-null; loaded under `guard`.
            if table.is_null() || unsafe { table.deref() }.bins.len() == 0 {
                table = self.init_table(guard);
                continue;
            }

            // SAFETY: only dereferenced when non-null; loaded under `guard`.
            let bini = unsafe { table.deref().bin_index(h) };
            // SAFETY: only dereferenced when non-null; loaded under `guard`.
            let mut bin = unsafe { table.deref().bin(bini, guard) };
            if bin.is_null() {
                // fast path, bin empty, so stick at front

                // SAFETY: only dereferenced when non-null; loaded under `guard`.
                unsafe {
                    match table.deref().cas_bin(bini, bin, node, guard) {
                        Ok(_old_null_ptr) => {
                            self.add_count(1, Some(0), guard);
                            return None;
                        }
                        Err(changed) => {
                            assert!(!changed.current.is_null());
                            node = changed.new;
                            bin = changed.current;
                        }
                    }
                }
            }

            // slow path,  bin is non-empty, need to link into it

            match *bin {
                BinEntry::Moved(next_table) => {
                    table = self.help_transfer(table, next_table, guard);
                    continue;
                }
                BinEntry::Node(ref head)
                    if no_replacement && head.hash == h && &head.key == key =>
                {
                    //fast path if replacement is disallowed and first bin matches
                    return Some(());
                }
                BinEntry::Node(ref head) => {
                    let head = Shared::from(head as *const _);
                    // bin is non-empty, need to link into it, so we must take lock
                    let _guard = head.lock.lock();

                    // need to check this is still the head
                    let current_head = table.bin(bini, guard);
                    if current_head.as_raw() != bin.as_raw() {
                        // nope -- try again from the start
                        continue;
                    }

                    //yes, it is still in the head, so we can now "own" the bin
                    //note that there can still be readers in the bin!

                    //TODO: TreeBin & ReservationNode

                    let mut bin_count = 0;
                    let mut n = head;

                    let old_val = loop {
                        if n.hash == hash && n.key == key {
                            // the key already exist in the map
                            if no_replacement {
                                // the key is not absent, so don't update
                            } else if let BinEntry::Node(Node { value, .. }) = *node.into_box() {
                                // we want to update
                                let now_garbage = n.value.swap(value, Ordering::SeqCst, guard);
                                unimplemented!("need to dispose of garbage");
                            } else {
                                unreachable!();
                            }
                            break Some(());
                        }

                        //TODO: ordering can probalby be relaxed due to mutex
                        let next = n.next.load(Ordering::SeqCst, guard);
                        if next.is_null() {
                            // at end of bin, stick node here
                            n.next.store(node, Ordering::SeqCst);
                            break None;
                        }
                        n = next;
                        bin_count += 1;
                    };

                    // TODO: TREEIFY_TRESHOLD

                    if old_val.is_none() {
                        //increment count
                        self.add_count(1, Some(bin_count));
                    }
                    return old_val;
                }
            }
        }
    }

    fn init_table(&self, guard: &Guard) -> Shared<Table<K, V>> {
        loop {
            // try to load table or be the one to init it
            let table = self.table.load(Ordering::SeqCst, guard);
            if !table.is_null() && table.bins.is_empty() {
                break table;
            }
            // try to allocate table
            let mut sc = self.size_ctl.load(Ordering::SeqCst);
            if sc < 0 {
                // we lost the initialization race, spin
                std::thread::yield_now();
                continue;
            }

            // try to become the initializer
            if self
                .size_ctl
                .compare_exchange(sc, -1, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                // we got to it;
                //check that initialization is still needed.
                let mut table = self.table.load(Ordering::SeqCst, guard);
                if !table.is_null() || table.bins.is_empty() {
                    let n = if sc > 0 {
                        sc as usize
                    } else {
                        DEFAULT_CAPACITY
                    };
                    let new_table = Owned::new(Table {
                        bins: vec![Atomic::null(); n].into_boxed_slice(),
                    });
                    table = new_table.into_shared(guard);
                    self.table.store(table, Ordering::SeqCst);
                    sc = n as isize - (n >> 2) as isize;
                }
                // "releasing" the lock
                self.size_ctl.store(sc, Ordering::SeqCst);
                break table;
            }
        }
    }

    fn add_count(&self, n: isize, resize_hint: Option<usize>, guard: &Guard) {
        //TODO: implement the java counter cell business here

        let count = if n > 0 {
            let n = n as usize;
            self.count.fetch_add(n, Ordering::SeqCst) + n
        } else if n < 0 {
            let n = n.abs() as usize;
            self.count.fetch_sub(n, Ordering::SeqCst) - n
        } else {
            self.count.load(Ordering::SeqCst)
        };

        // if resize_hint is None, it means that the caller does not want us to consider a resize
        // if it is Some(n), the caller saw n entries in a bin
        if resize_hint.is_none() {
            return;
        }

        let saw_bin_length = resize_hint.unwrap();

        loop {
            let sc = self.sc.load(Ordering::SeqCst);

            // when positive sc is the next resize treshold, so if count < sc we are nto resizing
            // if resizing, sc is negative, and then less than count, not triggering the branch
            if (count as isize) < sc {
                // we're not at the next resize point yet
                break;
            }

            let mut table = self.table.load(Ordering::SeqCst, guard);
            if table.is_null() {
                // table has not been initialized by another thread anyway
                break;
            }

            let n = table.bins.len();
            if table.bins.len() >= MAXIMUM_CAPACITY {
                // cant rezise anymore anyway
                break;
            }

            let rs = Self::resize_stamp(n) << RESIZE_STAMP_SHIFT;

            // Size control is negative means that a resize is happening, try to join if we are
            // allowed to assist
            if sc < 0 {
                // ongoing resize! can we join the resize transfer?
                if sc == rs + MAX_RESIZERS || sc == rs + 1 {
                    break;
                }
                let nt = self.next_table.load(Ordering::SeqCst, guard);
                if nt.is_null() {
                    break;
                }
                if self.transfer_index.load(Ordering::SeqCst) <= 0 {
                    break;
                }

                // try to join resize
                if self
                    .size_ctl
                    .compare_exchange(sc, sc + 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    self.transfer(table, nt, guard);
                }
            } else if self
                .size_ctl
                .compare_exchange(sc, rs + 2, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                // a resize is needed, but has not yet started
                // TODO: figure out why this is rs + 2, not just rs
                self.transfer(table, Shared::null(), guard);
            }

            // Another resize may be needed!
            count = self.count.load(Ordering::SeqCst);
        }
    }

    fn transfer(&self, table: Shared<Table>, mut next_table: Shared<Table>, guard: &Guard) {
        let n = table.bins.len();
        //TODO: use num_cpus to determine stride
        let stride = MIN_TRANSFER_STRIDE;

        if next_table.is_null() {
            // we are initiating a new resize
            let table = Owned::new(Table {
                bins: vec![Atomic::null(); n << 1].into_boxed_slice(),
            });

            // NOTE: we know that we "own" this resize, due to the CAS of sc in the add_count
            let now_garbage = self.next_table.swap(table, Ordering::SeqCst, guard);
            assert!(now_garbage.is_null());
            self.transfer_index.store(n, Ordering::SeqCst);
            next_table = self.next_table.load(Ordering::Relaxed, guard);
        }

        let next_n = next_table.bins.len();
        let mut advance = true;
        let mut finishing = false;
        let mut i = 0;
        let mut bound = 0;
        loop {
            //try to claim a range of bins for us to transfer
            while advance {
                i -= 1;

                if i >= bound || finishing {
                    advance = false;
                    break;
                }

                let next_index = self.transfer_index.load(Ordering::SeqCst);
                if next_index <= 0 {
                    // already finnished
                    i = -1;
                    advance = false;
                    break;
                };

                let next_bound = if next_index > stride as isize {
                    next_index - stride as isize
                } else {
                    0
                };
                if self
                    .transfer_index
                    .compare_exchange(next_index, next_bound, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    // we get to transfer everything from i to bound
                    bound = next_bound;
                    i = next_index - 1;
                    advance = false;
                    break;
                }
            }

            if i < 0 || i >= n || i + n >= next_n {
                //resize has finnished
                if finishing {
                    // this branch is only taken for one thread partaking in the resize!
                    self.next_table.store(Shared::null(), Ordering::SeqCst);
                    //old table that we are now finnished moving stuff out off
                    let now_garbage = self.table.swap(next_table, Ordering::SeqCst, guard);
                    unimplemented!("do something with garbage");
                    self.size_ctl.store((n << 1) - (n >> 1), Ordering::SeqCst);
                    return;
                }

                let sc = self.size_ctl.load(Ordering::SeqCst);
                if self
                    .size_ctl
                    .compare_exchange(sc, sc - 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    if (sc - 2) != Self::resize_stamp(n) << RESIZE_STAMP_SHIFT {
                        return;
                    }

                    // We are the choosen thread to finnish resize
                    finishing = true;

                    // we will also check if we can assist with any subsequent resize
                    // WHY: will break next loop anyway
                    advance = true;

                    // NOTE:the java code says "recheck before commit" here
                    i = n;
                }

                continue;
            }

            let bin = table.bin(i as usize, guard);
            if bin.is_null() {
                //TODO: re-use BinEntry::Moved where possible
                advance = table
                    .cas_bin(
                        i,
                        Shared::null(),
                        Owned::new(BinEntry::Moved(next_table.as_raw())),
                        guard,
                    )
                    .is_ok();
                continue;
            }

            match *bin {
                BinEntry::Moved(_) => {
                    //already processed
                    advance = true;
                }
                BinEntry::Node(ref head) => {
                    let head = Shared::from(head as *const _);
                    // bin is non-empty, need to link into it, so we must take lock
                    let _guard = head.lock.lock();

                    // need to check this is still the head
                    let current_head = table.bin(i, guard);
                    if current_head.as_raw() != bin.as_raw() {
                        // nope -- try again from the start
                        continue;
                    }

                    //yes, it is still in the head, so we can now "own" the bin
                    //note that there can still be readers in the bin!

                    //TODO: Treebin & ReservationNode

                    let mut run_bit = head.hash & n as u64;
                    let mut last_run = head;

                    let mut p = head;
                    while !p.is_null() {
                        let node = unsafe { p.deref() };
                        let b = node.hash & n as u64;
                        if b != run_bit {
                            run_bit = b;
                            last_run = p;
                        }
                        p = node.next.load(Ordering::SeqCst, guard);
                    }

                    let mut low_bin = Shared::null();
                    let mut high_bin = Shared::null();
                    if run_bit == 0 {
                        // last run is all in the low bin
                        low_bin = last_run;
                    } else {
                        //last run is all in the high bin
                        high_bin = last_run;
                    }

                    p = head;
                    while p != last_run {
                        let link = if p.hash & n as u64 == 0 {
                            &mut low_bin
                        } else {
                            &mut high_bin
                        };

                        // link is moved into next, and next becomes the new struct.
                        *link = Owned::new(Node {
                            hash: p.hash,
                            key: p.key.clone(),
                            lock: p.lock.clone(),
                            value: p.value.clone(),
                            next: Atomic::from(*link),
                        })
                        .into_shared(guard);
                        //WHY: what is happening here with owned -> share etc

                        p = p.next.load(Ordering::SeqCst, guard);
                    }

                    next_table.store_bin(i, low_bin);
                    next_table.store_bin(i + n, high_bin);
                    table.store_bin(i, Owned::new(BinEntry::Moved(next_table.as_raw())));
                    advance = true;
                }
            }
        }
    }

    /// Helps transfer if a resize is in progress.
    fn help_transfer<'g>(
        &'g self,
        table: Shared<'g, Table<K, V>>,
        next_table: *const Table<K, V>,
        guard: &'g Guard,
    ) -> Shared<'g, Table<K, V>> {
        if table.is_null() || next_table.is_null() {
            // NOTE: java returns the current `table` field here, not the `tab` argument
            return self.table.load(Ordering::SeqCst, guard);
        }

        let next_table = Shared::from(next_table);

        // SAFETY: `table` is non-null, and tables are only reclaimed through
        // epoch-deferred destruction, so it stays valid while `guard` is pinned.
        let n = unsafe { table.deref() }.bins.len();
        let rs = Self::resize_stamp(n as isize) << RESIZE_STAMP_SHIFT;

        while next_table == self.next_table.load(Ordering::SeqCst, guard)
            && table == self.table.load(Ordering::SeqCst, guard)
        {
            let sc = self.size_ctl.load(Ordering::SeqCst);
            if sc >= 0
                || sc == rs + MAX_RESIZERS as isize
                || sc == rs + 1
                || self.transfer_index.load(Ordering::SeqCst) <= 0
            {
                // not resizing, resize is finishing, or no more ranges to claim
                break;
            }

            // try to join the resize
            if self
                .size_ctl
                .compare_exchange(sc, sc + 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                self.transfer(table, next_table, guard);
                break;
            }
        }
        next_table
    }
    /// Returns the stamp bits for resizing a table of size n.
    /// Must be negative when shifted left by RESIZE_STAMP_SHIFT.
    fn resize_stamp(n: isize) -> isize {
        return n.leading_zeros() as isize | (1 << (RESIZE_STAMP_BITS - 1));
    }
}

struct Table<K, V> {
    bins: Box<[Atomic<node::BinEntry<K, V>>]>,
}

impl<K, V> Table<K, V> {
    #[inline]
    fn bini(&self, hash: u64) -> usize {
        // We take the low bits of the hash that can index into the amount of bins we have
        //
        // 0b101010101010100110
        //
        //  4 bins == 0b100
        //  mask = (4-1) == 0b011
        //
        //  hash & mask = 0b011 & 0b.....010 -> 0b10
        let mask = self.bins.len() as u64 - 1;
        (hash & mask) as usize
    }

    /// If you have a reference to a table allready, you can get a node.
    #[inline]
    fn bin<'g>(&'g self, i: usize, guard: &'g Guard) -> Shared<'g, node::BinEntry<K, V>> {
        self.bins[i].load(Ordering::Acquire, guard)
    }

    /// if succeeds we need to que old for garbage collection,
    /// if fails, owned still has to be dealed with
    #[inline]
    fn cas_bin<'g>(
        self,
        i: usize,
        current: Shared<node::BinEntry<K, V>>,
        new: Owned<node::BinEntry<K, V>>,
        guard: &'g Guard,
    ) -> Result<
        Shared<'g, node::BinEntry<K, V>>,
        crossbeam::epoch::CompareAndSetError<'g, node::BinEntry<K, V>, Owned<node::BinEntry<K, V>>>,
    > {
        self.bins[i].compare_and_set(current, new, Ordering::AcqRel, guard)
    }

    #[inline]
    fn store_bin(self, i: usize, new: Owned<node::BinEntry<K, V>>) {
        self.bins[i].store(new, Ordering::Release);
    }
}
