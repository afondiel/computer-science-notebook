
///////////////////////////////////////////////////////
// design-patterns-for-embedded-systems-in-c
// by : Bruce Powel Douglass - PhD
///////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "HW_ProxyPattern\include\proxy_pattern.h"
#include "main_hw.h"

// #83 : Manipulating bit-oriented memory mapped hardware
#define TURN_OFF (0x00)
#define INITIALIZE (0x61)
#define RUN (0x69)
#define CHECK_ERROR (0x02)
#define DEVICE_ADDRESS (0x01FFAFD0)

void emergencyShutDown(void){
    printf("OMG We’re all gonna die!\n");
}
// ------------------------------------------- //


void main_hw_access(void)
{
    printf("======= main_hw_access =========\n");

    // Manipulating bit-oriented memory mapped hardware
    unsigned char* pDevice;
    pDevice = (unsigned char *)DEVICE_ADDRESS; // pt to device
    // for testing you can replace the above line with
    // pDevice = malloc(1);

    *pDevice = 0xFF; // start with all bits on
    printf ("Device bits %X\n", *pDevice);

    *pDevice = *pDevice & INITIALIZE; // and the bits into
    printf ("Device bits %X\n", *pDevice);

    if (*pDevice & CHECK_ERROR) { // system fail bit on?
        emergencyShutDown();
        abort();
    }
    else {
        *pDevice = *pDevice & RUN;
    }

}
