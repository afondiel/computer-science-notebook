#include <stdio.h>
#include <stdlib.h>
#include "../include/queue_OO.h"

void Queue_Init(    Queue* const me,
                    int (*isFullfunction)(Queue* const me),
                    int (*isEmptyfunction)(Queue* const me),
                    int (*getSizefunction)(Queue* const me),
                    void (*insertfunction)(Queue* const me, int k),
                    int (*removefunction)(Queue* const me)
                )
{
    /* initialize attributes */
    me->q_head = 0;
    me->q_size = 0;
    me->q_tail = 0;

    /* initialize member function pointers */
    me->isFull    = isFullfunction;
    me->isEmpty   = isEmptyfunction;
    me->getSize   = getSizefunction;
    me->q_insert  = insertfunction;
    me->q_remove  = removefunction;
}

/* operation Cleanup() */
void Queue_Cleanup(Queue* const me) {
    //nothing to do
}

/* operation isFull() */
int Queue_isFull(Queue* const me){
    return (me->q_head+1) % QUEUE_SIZE == me->q_tail;
}

/* operation isEmpty() */
int Queue_isEmpty(Queue* const me){
    return (me->q_head == me->q_tail);
}
/* operation getSize() */
int Queue_getSize(Queue* const me) {
    return me->q_size;
}
/* operation insert(int) */
void Queue_insert(Queue* const me, int k) {
    if (!me->isFull(me)) {
        me->buffer[me->q_head] = k;
        me->q_head = (me->q_head+1) % QUEUE_SIZE;
        ++(me->q_size);
    }
}

/* operation remove */
int Queue_remove(Queue* const me) {
    int value = -9999; /* sentinel value */

    if (!me->isEmpty(me)) {
        value = me->buffer[me->q_tail];
        me->q_tail = (me->q_tail+1) % QUEUE_SIZE;
        --(me->q_size);
    }
    return value;
}

Queue * Queue_Create(void) {
    Queue* me = (Queue *) malloc(sizeof(Queue));
    if(me!=NULL)
    {
        Queue_Init(me, Queue_isFull, Queue_isEmpty, Queue_getSize, Queue_insert, Queue_remove);
    }
    return me;
}
void Queue_Destroy(Queue* const me) {
    if(me!=NULL)
    {
        Queue_Cleanup(me);
    }
    free(me);
}
