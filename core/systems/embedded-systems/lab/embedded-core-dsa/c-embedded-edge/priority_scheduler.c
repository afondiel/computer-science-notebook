#include <stdint.h>

#define MAX_TASKS 10

typedef struct {
    void (*task)(void);
    uint8_t priority;
} Task;

typedef struct {
    Task tasks[MAX_TASKS];
    uint8_t size;
} PriorityScheduler;

void scheduler_init(PriorityScheduler* sched) {
    sched->size = 0;
}

void scheduler_add(PriorityScheduler* sched, void (*task)(void), uint8_t prio) {
    if(sched->size >= MAX_TASKS) return;
    
    // Insertion sort by priority
    int i = sched->size - 1;
    while(i >= 0 && sched->tasks[i].priority > prio) {
        sched->tasks[i+1] = sched->tasks[i];
        i--;
    }
    sched->tasks[i+1] = (Task){task, prio};
    sched->size++;
}

void scheduler_run(PriorityScheduler* sched) {
    if(sched->size > 0) {
        sched->tasks[0].task();
        for(int i=0; i<sched->size-1; i++) {
            sched->tasks[i] = sched->tasks[i+1];
        }
        sched->size--;
    }
}

int main(void)
{
    PriorityScheduler sched;
    scheduler_init(&sched);
    
    void task1(void) {
        printf("Task 1\n");
    }
    
    void task2(void) {
        printf("Task 2\n");
    }
    
    scheduler_add(&sched, task2, 1);
    scheduler_add(&sched, task1, 0);
    
    scheduler_run(&sched);
    scheduler_run(&sched);
    
    return 0;
}