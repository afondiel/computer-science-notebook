
///////////////////////////////////////////////////////
// design-patterns-for-embedded-systems-in-c
// by : Bruce Powel Douglass - PhD
///////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "ECRM_main.h"
//#include "4.1-Basic Concurrency Concepts\include"
//#include "4.2-Cyclic Executive Patterncepts\include"
//#include "4.3-Static Priority Pattern\include"
//#include "4.4-Critical Region Pattern\include"
//#include "4.5-Guarded Call Pattern\include"
//#include "4.6-Queuing Pattern\include"
//#include "4.7-Rendezvous Pattern\include"
//#include "4.8-Simultaneous Locking Pattern\include"
//#include "4.9-Ordered Locking\include"

#define TRUE (1)
/* ================ ECRM MAIN TEST ============================= */

/**
*@fn ECRM_main
*@brief
*@param
*@return
*/
void ECRM_main(void)
{
    printf("//-------- Design Patterns for Embedding Concurrency and Resource Management-------------//\n");


    printf("///////////////////////////\
    //4.1-Basic Concurrency Concepts //\
    //////////////////////////\n");

    /**==== Cyclic executive processing loop ====**/
    /* global static and stack data */
    static int nTasks = 3;
    int currentTask;
    /* initialization code */
    currentTask = 0;
    if (POST()) { /* Power On Self Test succeeds */
        /* scheduling executive */
        while (TRUE) {
            task1();
            task2();
            task3();
        }; /* end cyclic processing loop */
    }

    /** ====Basic task structure with a preemptive scheduler ==== **/
    /*here is where private static data goes*/
    static int taskAInvocationCounter = 0;
    void taskA(void) {
        /* more private task data */
        int stufftoDo;
        /* initialization code */
        stuffToDo = 1;
        while (stuffToDo) {
            signal = waitOnSignal();
            switch (signal) {
                case signal1:
                /* signal 1 processing here */
                break;
                case signal2:
                /* signal 2 processing here */
                case signal3:
                /* signal 3 processing here */
            };
        }; /* end infinite while loop */
    }; /* end task */

    printf("//////////////////////////\
    //4.2-Cyclic Executive Pattern //\
    //////////////////////////\n");
    //?

    printf("//////////////////////////\
    //4.3-Static Priority Pattern //\
    //////////////////////////\n");
    //?

    printf("///////////////////////////////\
    //4.4-Critical Region Pattern//\
    //////////////////////////////\n");

     printf("///////////////////////////////\
    //4.5-Guarded Call Pattern//\
    //////////////////////////////\n");

    printf("///////////////////////////////\
    //4.6-Queuing Pattern	//\
    //////////////////////////////\n");

    printf("///////////////////////////////\
    //4.7-Rendezvous Pattern	//\
    //////////////////////////////\n");

    printf("//////////////////////////////////////\
    //4.8-Simultaneous Locking Pattern	//\
    /////////////////////////////////////\n");

    printf("///////////////////////////////\
    //4.9-Ordered Locking	//\
    //////////////////////////////\n");



    //?

}
