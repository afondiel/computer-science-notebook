use core::sync::atomic::{AtomicUsize, Ordering};

pub struct CircularBuffer<T, const N: usize> {
    buffer: [core::mem::MaybeUninit<T>; N],
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T, const N: usize> CircularBuffer<T, N> {
    pub const fn new() -> Self {
        CircularBuffer {
            buffer: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub fn push(&self, item: T) -> Result<(), T> {
        let mut head = self.head.load(Ordering::Relaxed);
        let mut next = (head + 1) % N;
        
        if next == self.tail.load(Ordering::Acquire) {
            return Err(item);
        }

        match self.head.compare_exchange_weak(head, next, Ordering::Release, Ordering::Relaxed) {
            Ok(_) => {
                unsafe { self.buffer[head].as_mut_ptr().write(item) };
                Ok(())
            },
            Err(_) => self.push(item),
        }
    }

    pub fn pop(&self) -> Option<T> {
        let mut tail = self.tail.load(Ordering::Relaxed);
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }

        let next = (tail + 1) % N;
        match self.tail.compare_exchange_weak(tail, next, Ordering::Release, Ordering::Relaxed) {
            Ok(_) => {
                let item = unsafe { self.buffer[tail].as_ptr().read() };
                Some(item)
            },
            Err(_) => self.pop(),
        }
    }
}

// Usage
static BUFFER: CircularBuffer<i32, 5> = CircularBuffer::new();
