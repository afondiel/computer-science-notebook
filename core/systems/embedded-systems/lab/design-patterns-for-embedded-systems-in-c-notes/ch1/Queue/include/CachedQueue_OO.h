#ifndef CACHEDQUEUE_H_
#define CACHEDQUEUE_H_

/* include base class */
#include "..\include\queue_OO.h"

/* class CachedQueue */
typedef struct CachedQueue CachedQueue;
struct CachedQueue {

    /* base class */
    Queue* inqueue;
    /* new attributes */
    char filename[80];
    int numberElementsOnDisk;
    /* aggregation in subclass */
    Queue* outputQueue;

    /* inherited virtual functions */
    int (*isFull)(CachedQueue* const me);
    int (*isEmpty)(CachedQueue* const me);
    int (*getSize)(CachedQueue* const me);
    void (*q_insert)(CachedQueue* const me, int k);
    int (*q_remove)(CachedQueue* const me);
    /* new virtual functions */
    void (*q_flush)(CachedQueue* const me);
    void (*q_load)(CachedQueue* const me);

};

/* Constructors and destructors:*/
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
                );

void CachedQueue_Cleanup(CachedQueue* const me);

/* Operations */
int CachedQueue_isFull(CachedQueue* const me);
int CachedQueue_isEmpty(CachedQueue* const me);
int CachedQueue_getSize(CachedQueue* const me);
void CachedQueue_insert(CachedQueue* const me, int k);
int CachedQueue_remove(CachedQueue* const me);
void CachedQueue_flush(CachedQueue* const me);
void CachedQueue_load(CachedQueue* const me);

CachedQueue * CachedQueue_Create(void);
void CachedQueue_Destroy(CachedQueue* const me);

#endif /*CACHEDQUEUE_H_*/
