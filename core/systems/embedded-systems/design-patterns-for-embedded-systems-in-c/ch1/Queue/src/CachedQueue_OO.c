#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "..\include\queue_OO.h"
#include "..\include\cachedQueue_OO.h"

void CachedQueue_Init(
                        CachedQueue* const me,
                        char* fName,
                        int (*isFullfunction)(CachedQueue* const me),
                        int (*isEmptyfunction)(CachedQueue* const me),
                        int (*getSizefunction)(CachedQueue* const me),
                        void (*insertfunction)(CachedQueue* const me, int k),
                        int (*removefunction)(CachedQueue* const me),
                        void (*flushfunction)(CachedQueue* const me),
                        void (*loadfunction)(CachedQueue* const me)
                      )
{
    /* initialize base class */
    me->inqueue = Queue_Create(); /* queue member must use its original functions */
    /* initialize subclass attributes */
    me->numberElementsOnDisk = 0;

    strcpy(me->filename, fName);

    /* initialize aggregates */
    me->outputQueue = Queue_Create();
    /* initialize subclass virtual operations ptrs */
    me->isFull = isFullfunction;
    me->isEmpty = isEmptyfunction;
    me->getSize = getSizefunction;
    me->q_insert = insertfunction;
    me->q_remove = removefunction;
    me->q_flush = flushfunction;
    me->q_load = loadfunction;
}
/* operation Cleanup() */
void CachedQueue_Cleanup(CachedQueue* const me) {
    Queue_Cleanup(me->inqueue);
}
/* operation isFull() */
int CachedQueue_isFull(CachedQueue* const me){
    return me->inqueue->isFull(me->inqueue) && me->outputQueue->isFull(me->outputQueue);
}
/* operation isEmpty() */
int CachedQueue_isEmpty(CachedQueue* const me){
    return me->inqueue->isEmpty(me->inqueue) &&
    me->outputQueue->isEmpty(me->outputQueue) &&
    (me->numberElementsOnDisk == 0);
}

/* operation getSize() */
int CachedQueue_getSize(CachedQueue* const me) {
    return me->inqueue->getSize(me->inqueue) +
    me->outputQueue->getSize(me->outputQueue) +
    me->numberElementsOnDisk;
}
/* operation q_insert(int) */
// Imsert Algorithm:
// if the queue is full,
// call flush to write out the queue to disk and reset the queue
// end if
// insert the data into the queue
void CachedQueue_insert(CachedQueue* const me, int k) {
    if (me->inqueue->isFull(me->inqueue)) me->q_flush(me);
    me->inqueue->q_insert(me->inqueue, k);
}
/* operation remove */
// remove algorithm
// if there is data in the outputQueue,
// remove it from the outputQueue

// else if there is data on disk
// call load to bring it into the outputQueue
// remove it from the outputQueue
// else if there is data in the queue
// remove it from there
// (if there is no data to remove then return sentinel value)
int CachedQueue_remove(CachedQueue* const me) {

    if (!me->outputQueue->isEmpty(me->outputQueue))
        return me->outputQueue->q_remove(me->outputQueue);
    else if (me->numberElementsOnDisk>0) {
        me->q_load(me);
        return me->inqueue->q_remove(me->inqueue);
    }
    else
        return me->inqueue->q_remove(me->inqueue);
}
/* operation flush */
// Precondition: this is called only when queue is full
// and filename is valid
// flush algorithm
// if file is not open, then open file
// while not queue->isEmpty()
// queue->remove()
// write data to disk
// numberElementsOnDisk++
// end while
void CachedQueue_flush(CachedQueue* const me){
// write file I/O statements here . . .
}


/* operation load */
// Precondition: this is called only when outputQueue is empty
// and filename is valid
// load algorithm
// while (!outputQueue->isFull() && (numberElementsOnDisk>0)
// read from start of file (i.e., oldest datum)
// numberElementsOnDisk– –;
// outputQueue->q_insert()
// end while
void CachedQueue_load(CachedQueue* const me) {
// write file I/O statements here . . .
}

CachedQueue * CachedQueue_Create(void) {

    CachedQueue* me = (CachedQueue *) malloc(sizeof(CachedQueue));
    if(me!=NULL)
    {
        CachedQueue_Init(me, "queuebuffer.dat", CachedQueue_isFull,
                         CachedQueue_isEmpty, CachedQueue_getSize, CachedQueue_insert,
                         CachedQueue_remove, CachedQueue_flush, CachedQueue_load
                         );
    }
    return me;
}

void CachedQueue_Destroy(CachedQueue* const me) {

    if(me!=NULL)
    {
        CachedQueue_Cleanup(me);
    }
    free(me);
}

