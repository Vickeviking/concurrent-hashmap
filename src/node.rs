use crossbeam::epoch::Atomic;
use std::cell::UnsafeCell;

/// Entry in a bin.
///
/// Will generally be `Node`. Any entry that is not first in the bin, will be a `Node`.
pub(crate) enum BinEntry<K, V> {
    Node(Node<K, V>),
}

/// Key-value entry.  
pub(crate) struct Node<K, V> {
    hash: u64,
    key: K,
    value: UnsafeCell<V>,
    next: Atomic<Node<K, V>>,
}
