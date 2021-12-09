
///////////////////////////////////////////////////////
// design-patterns-for-embedded-systems-in-c
// by : Bruce Powel Douglass - PhD
///////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "main_hw.h"
#include "3-1 - HW accessing concept\hw_test.h"
#include "3-2 - HW_ProxyPattern\include\HWProxyExample.h"



/* ================ HW MAIN TEST ============================= */

void main_hw_access(void)
{
    printf("//------------------------- main_hw_access -------------------------//\n");

    //Test bitfielf_TEST1 :
    /* \Warning: change DEVICE_ADDRESS value to avoid segmentation fault
     *
     */
    bitField_TEST1();

    //Test bitfielf_TEST2 :
    /*
     */
    //bitField_TEST2();

    //////////////////////////
    //HW Proxy example test //
    //////////////////////////
    //?

    //////////////////////////
    //HW Adapter example test //
    //////////////////////////
    //?

    //////////////////////////
    //HW Mediator example test //
    //////////////////////////
    //?


}
