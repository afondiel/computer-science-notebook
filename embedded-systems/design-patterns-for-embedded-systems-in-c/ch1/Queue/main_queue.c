
///////////////////////////////////////////////////////
// design-patterns-for-embedded-systems-in-c
// by : Bruce Powel Douglass - PhD
///////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "include\queue_OO.h"
#include "include\CachedQueue_OO.h"
#include "main_queue.h"

void main_queue(void)
{
    //int ret_OK = 0;
//    int j,k, h, t;
//
//    /* test normal queue */
//    Queue * myQ;
//    myQ = Queue_Create();
//    k = 1000;
//
//    for (j=0;j<QUEUE_SIZE;j++) {
//        h = myQ->q_head;
//        myQ->q_insert(myQ,k);
//        printf("inserting %d at position %d, size =%d\n",k--,h, myQ->getSize(myQ));
//    }
//
//    printf("Inserted %d elements\n",myQ->getSize(myQ));
//
//    for (j=0;j<QUEUE_SIZE;j++) {
//        t = myQ->q_tail;
//        k = myQ->q_remove(myQ);
//        printf("REMOVING %d at position %d, size =%d\n",k,t, myQ->getSize(myQ));
//    }
//
//    printf("Last item removed = %d\n", k);
//    printf("Current queue size %d\n", myQ->getSize(myQ));
//    puts("Queue test program");
//
    //return EXIT_SUCCESS;
    //return ret_OK;

    printf("Cached Queue TEST \n");

}

