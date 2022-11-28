#include <stdio.h>
#include <stdlib.h>

// *scheduler Notes

/* Types :
    - FCFS (No priority)
    - SJF (Short Job First)
    - priority Scheduling
    - Round Robin (non-premptive)

    Non-premptive scheduler features :
    * The ability of a task to halt(stop) itself.
    * The ability of a task to sleep for a while

    ** Tasks state : **
    - ready
    - waiting
    - inactive

*/

#define NUMBER_OF_FUNCTIONS 10

//an array of function pointers to check the task states
void (*list_of_functions[NUMBER_OF_FUNCTIONS])(void *p);
void taskA(void *p);
void taskB(void *p);
void taskC(void *p);
void time_delay(void);
void start_function(void (*functionPTR)() );
// function prototype for scheduler
void scheduler();

int main()
{
    //while(1)
    //{
        //old 1 :
        //taskA();
        //taskB();
        //taskC();
        //time_delay();

        //old 2 :
        //start_function( taskA);
        //start_function( taskB);
        //start_function( taskC);
        //time_delay();
    //}
    //haltedtasks[]
    // initialize array of pointers to tasks
    readytasks[0] = taskA;
    readytasks[1] = taskB;
    readytasks[2] = taskC;
    readytasks[3]= NULL; // NULL signals the last task in the list.
    // now start scheduler
    while(1) {
        scheduler();
        time_delay();  //tune or use timer for 1ms loop time
    }

}


//Tasks
void taskA(void)
{
   printf("Bonjour");
}
void taskB(void)
{
   printf("Salut");
}
void taskC(void)
{
   printf(" Hello ");
}
void time_delay(void)
{
   int cnt = 0.1;
   while(!cnt)
        cnt--;
}

//function pointer
void start_function(void (*functionPTR)() )
{
    functionPTR();
}

void scheduler()
{
    int task_index;
    if(readytasks[task_index] == NULL && task_index != 0)
        task_index=0;
    if(readytasks[task_index] == NULL && task_index == 0) {
        // figure out something to do because there are no tasks to run!
    }
    start_function(readytasks[task_index]);
    task_index++; // Round Robin/we’re taking turns

//    * for each element of waitingtasks which is not NULL:
//    decrement the delay value d.
//    if (d==0) move the function pointer to the end
//    of the readytasks[] array and remove it
//    from waitingtasks.
//    // end of scheduler

    return;
}

void halt_me()
{
//    * identify which task is currently running (i.e. look at task_index)
//    * copy the function pointer from readytasks[task_index]
//    to the haltedtasks array
//    * move the remaining tasks up in readytasks[] to fill the
//    empty hole and copy NULL into the last element.
//    * increment the index of the haltedtasks array.
    return;
}


Sleep(int d)
{
//    * copy function pointer from readytasks[task_index]
//to the waitingtasks[] array.
//* clean up readytasks[] as in halt_me();
//* copy the d into the delays array with the same index
//as the function pointer has in waitingtasks[]
    return;
}

//task using sleep
task_c(p) {
    compute for a while;
    sleep(10);
    return;
}


//=============== //Alternate Data Structure for Scheduling : instead of ARRAYS=================//
//Task states
#define STATE_RUNNING   0
#define STATE_READY     1
#define STATE_WAITING   2
#define STATE_INACTIVE  3

//Task Control Blocks (TCB)
typedef struct TCBstruct {
    void (*ftpr)(void *p);          // the function pointer
    void *arg_ptr;                  // the argument pointer
    unsigned short int state;       // the task state
    unsigned int delay;             // sleep delay
} tcb;

//Static
tcb TaskList[N_MAX_TASKS];

//Prototypes functions
start_task(int task_id);
halt_me();
delay(int d);

int main()
{
    int j=0;
    int task_B_Arg;

    TaskList[j].ftpr = task_A();
    TaskList[j].arg_ptr = NULL;
    TaskList[j].state = STATE_INACTIVE;
    TaskList[j].delay = -1;
    j++;
    TaskList[j].ftpr = task_B();
    task_B_Arg = 56; // some arbitrary value
    int *ip = &task_B_Arg;
    TaskList[j].arg_ptr = (void*)ip;
    TaskList[j].state = STATE_READY;
    TaskList[j].delay = -1;
    j++;
    TaskList[j].fptr = NULL;

    return 0;
}


//functions
halt_me() {
    TaskList[t_curr].state = STATE_INACTIVE;
}
start_task(int task_id) {
    TaskList[task_id].state = STATE_READY;
}
delay(int d) {
    TaskList[t_curr].delay = d;
    TaskList[t_curr].state = STATE_WAITING;
}
7
