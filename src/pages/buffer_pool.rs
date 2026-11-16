// SPDX-License-Identifier: Apache-2.0

//! Reuse decompression allocations only after their last page or array owner is released.

use bytes::Bytes;
use parking_lot::Mutex;

const MIN_BUFFER_BYTES: usize = 64 * 1024;
const MAX_BUFFER_BYTES: usize = 1024 * 1024;
const MAX_BUFFERS: usize = 16;

// At most 16 MiB of released allocations. Contention falls back to the ordinary allocator.
static RELEASED: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

pub(super) struct PageBuffer(pub(super) Vec<u8>);

impl PageBuffer {
    pub(super) fn new(size: usize) -> Self {
        if (MIN_BUFFER_BYTES..=MAX_BUFFER_BYTES).contains(&size)
            && let Some(mut buffers) = RELEASED.try_lock()
            && let Some(index) = buffers
                .iter()
                .enumerate()
                .filter(|(_, buffer)| buffer.capacity() >= size)
                .min_by_key(|(_, buffer)| buffer.capacity())
                .map(|(index, _)| index)
        {
            return Self(buffers.swap_remove(index));
        }
        Self(Vec::with_capacity(size))
    }

    pub(super) fn into_bytes(mut self) -> Bytes {
        if (MIN_BUFFER_BYTES..=MAX_BUFFER_BYTES).contains(&self.0.capacity()) {
            // Bytes retains this owner until all clones and slices have been released.
            Bytes::from_owner(self)
        } else {
            Bytes::from(std::mem::take(&mut self.0))
        }
    }
}

impl AsRef<[u8]> for PageBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Drop for PageBuffer {
    fn drop(&mut self) {
        if (MIN_BUFFER_BYTES..=MAX_BUFFER_BYTES).contains(&self.0.capacity())
            && let Some(mut buffers) = RELEASED.try_lock()
            && buffers.len() < MAX_BUFFERS
        {
            self.0.clear();
            buffers.push(std::mem::take(&mut self.0));
        }
    }
}
