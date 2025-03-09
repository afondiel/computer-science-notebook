use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ptr::NonNull;

pub struct MemoryPool<T, const N: usize> {
    storage: [UnsafeCell<MaybeUninit<T>>; N],
    free_list: UnsafeCell<Option<NonNull<FreeListNode>>>,
}

struct FreeListNode {
    next: Option<NonNull<FreeListNode>>,
}

impl<T, const N: usize> MemoryPool<T, N> {
    pub const fn new() -> Self {
        MemoryPool {
            storage: unsafe { MaybeUninit::uninit().assume_init() },
            free_list: UnsafeCell::new(None),
        }
    }

    pub fn allocate(&self) -> Option<&mut T> {
        unsafe {
            let mut current = *self.free_list.get();
            if let Some(node) = current.take() {
                *self.free_list.get() = node.as_ref().next;
                let ptr = node.as_ptr() as *mut T;
                Some(&mut *ptr)
            } else {
                None
            }
        }
    }

    pub fn deallocate(&self, ptr: &mut T) {
        unsafe {
            let node = ptr as *mut T as *mut FreeListNode;
            let mut new_head = NonNull::new_unchecked(node);
            (*new_head.as_mut()).next = *self.free_list.get();
            *self.free_list.get() = Some(new_head);
        }
    }
}

unsafe impl<T, const N: usize> Sync for MemoryPool<T, N> {}

// Usage
static POOL: MemoryPool<i32, 2> = MemoryPool::new();
