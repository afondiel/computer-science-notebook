import heapq

class TaskScheduler:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, priority):
        heapq.heappush(self.tasks, (priority, func))

    def run_next(self):
        if self.tasks:
            _, task = heapq.heappop(self.tasks)
            task()

# Example usage
def high_priority_task():
    print("High priority task executed")

def low_priority_task():
    print("Low priority task executed")

scheduler = TaskScheduler()
scheduler.add_task(low_priority_task, 10)
scheduler.add_task(high_priority_task, 1)

scheduler.run_next()  # Executes high_priority_task
scheduler.run_next()  # Executes low_priority_task
