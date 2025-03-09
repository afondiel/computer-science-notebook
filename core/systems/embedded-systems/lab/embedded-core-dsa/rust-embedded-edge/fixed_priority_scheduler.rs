use heapless::Vec;
use core::cmp::Ordering;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Task {
    id: u8,
    priority: u8,
}

impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Task {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
    }
}

pub struct Scheduler {
    tasks: Vec<Task, 10>,
}

impl Scheduler {
    pub fn new() -> Self {
        Scheduler { tasks: Vec::new() }
    }

    pub fn add_task(&mut self, id: u8, priority: u8) -> Result<(), ()> {
        let task = Task { id, priority };
        self.tasks.push(task).map_err(|_| ())
    }

    pub fn get_next_task(&mut self) -> Option<u8> {
        if let Some(task) = self.tasks.pop() {
            Some(task.id)
        } else {
            None
        }
    }
}

// Usage
let mut scheduler: Scheduler = Scheduler::new();
scheduler.add_task(1, 10).unwrap();
scheduler.add_task(2, 5).unwrap();
assert_eq!(scheduler.get_next_task(), Some(1));
