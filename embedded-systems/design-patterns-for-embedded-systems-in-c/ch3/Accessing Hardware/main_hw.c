
///////////////////////////////////////////////////////
// design-patterns-for-embedded-systems-in-c
// by : Bruce Powel Douglass - PhD
///////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "HW_ProxyPattern\include\HWProxyExample.h"
#include "main_hw.h"

// #83 : Manipulating bit-oriented memory mapped hardware
#define TURN_OFF (0x00)
#define INITIALIZE (0x61)
#define RUN (0x69)
#define CHECK_ERROR (0x02)
#define DEVICE_ADDRESS (0x01FFAFD0)


// --------------------Functions definitions----------------------- //

// Manipulating bit-oriented memory mapped hardware
void emergencyShutDown(void)
{
    printf("OMG We’re all gonna die!\n");
}

void bitField_TEST1(void)
{
    unsigned char* pDevice;
    pDevice = (unsigned char *)DEVICE_ADDRESS; // pt to device
    // for testing you can replace the above line with
    // pDevice = malloc(1);

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

/* HW MAIN*/

void main_hw_access(void)
{
    printf("//------------------------- main_hw_access -------------------------//\n");

    //Test bitfielf_TEST1 :
    /* \Warning: change DEVICE_ADDRESS value to avoid segmentation fault
     *
     */
    //bitField_TEST1();

    //Test bitfielf_TEST2 :
    /*
     */
    bitField_TEST2();



}
