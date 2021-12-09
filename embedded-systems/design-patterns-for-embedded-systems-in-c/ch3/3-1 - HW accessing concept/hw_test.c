
///////////////////////////////////////////////////////
// design-patterns-for-embedded-systems-in-c
// by : Bruce Powel Douglass - PhD
///////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "hw_test.h"

// --------------------Functions definitions----------------------- //

// Manipulating bit-oriented memory mapped hardware
void emergencyShutDown(void)
{
    printf("OMG We’re all gonna die!\n");
}

void bitField_TEST1(void)
{
    unsigned char* pDevice;
    //pDevice = (unsigned char *)DEVICE_ADDRESS; // pt to device
    // for testing you can replace the above line with
    pDevice = malloc(1);  //Test purpose

    *pDevice = 0xFF; // start with all bits on
    printf("Device bits %X\n", *pDevice);

    *pDevice = *pDevice & INITIALIZE; // and the bits into
    printf("Device bits %X\n", *pDevice);

    if (*pDevice & CHECK_ERROR) { // system fail bit on?
        emergencyShutDown();
        abort();
    }
    else {
        *pDevice = *pDevice & RUN;
        printf("Device bits %X\n", *pDevice);
    }

}
void bitField_TEST2(void)
{
    typedef struct _statusBits {
        unsigned enable : 1;
        unsigned errorStatus : 1;
        unsigned motorSpeed : 4;
        unsigned LEDColor : 2;
    } statusBits;

    statusBits status;
    printf("size = %d\n",sizeof(status));
    status.enable = 1;
    status.errorStatus = 0;
    status.motorSpeed = 3;
    status.LEDColor = 2;

    if (status.enable) printf("Enabled\n");
    else printf ("Disabled\n");
    if (status.errorStatus) printf("ERROR!\n");
    else printf("No error\n");

    printf ("Motor speed %d\n",status.motorSpeed);
    printf ("Color %d\n",status.LEDColor);

}
// ------------------------------------------------------------------------------//
