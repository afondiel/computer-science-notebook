#ifndef MotorProxy_H
#define MotorProxy_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "HWProxyExample.h"
#include "MotorData.h"

/* class MotorProxy */
typedef struct MotorProxy MotorProxy;
/* This is the proxy for the motor hardware. */
/* Note that the speed of the motor is adjusted for the length of the rotary arm */
/* to keep a constant speed at the end of the arm. */
struct MotorProxy {
    unsigned int* motorAddr;
    unsigned int rotaryArmLength;
};

/* HW services :  setter//Getter//execute */
void MotorProxy_Init(MotorProxy* const me);
void MotorProxy_Cleanup(MotorProxy* const me);
DirectionType* MotorProxy_accessMotorDirection(MotorProxy* const me);
unsigned int MotorProxy_accessMotorSpeed(MotorProxy* const me);
unsigned int MotorProxy_aceessMotorState(MotorProxy* const me);

/* keep all settings the same but clear error bits */
void MotorProxy_clearErrorStatus(MotorProxy* const me);

/* Configure must be called first, since it sets up the */
/* address of the device. */
void MotorProxy_configure(MotorProxy* const me, unsigned int length, unsigned int* location);

/* turn motor off but keep original settings */
void MotorProxy_disable(MotorProxy* const me);

/* Start up the hardware but leave all other settings of the */
/* hardware alone */
void MotorProxy_enable(MotorProxy* const me);

/* precondition: must be called AFTER configure() function. */
/* turn on the hardware to a known default state. */
void MotorProxy_initialize(MotorProxy* const me);

/* update the speed and direction of the motor together */
void MotorProxy_writeMotorSpeed(MotorProxy* const me, const DirectionType* direction, unsigned int speed);

//Constractor
MotorProxy * MotorProxy_Create(void);
//Destractor
void MotorProxy_Destroy(MotorProxy* const me);


#endif

