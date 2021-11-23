///////////////////////////////////////////////////////
// design-patterns-for-embedded-systems-in-c
// by : Bruce Powel Douglass - PhD
///////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
//#include "ch1\Queue\include\queue_OO.h"
//#include "ch1\Queue\include\CachedQueue_OO.h"
//#include "ch1\Queue\main_queue.h"

//ch3
#include "ch3\Accessing Hardware\main_hw.h"
#include "ch3\Accessing Hardware\HW_ProxyPattern\include\HWProxyExample.h"
#include "ch3\Accessing Hardware\HW_ProxyPattern\include\MotorData.h"
#include "ch3\Accessing Hardware\HW_ProxyPattern\include\MotorProxy.h"

// Sensor exemples
/*Sensor mySensor = {
                   //Hz  //Hz //mV
                    100, 60, 5000
                  };

*/
///////////////////// MAIN TEST //////////////////////////
int main()
{
    printf(" = = = = = = = = = = Main function = = = = = = = = = = \n");

    // CONTENTS

    //printf("Chapter 1 :  What Is Embedded Programming?\n");

//Chapter 1 :  What Is Embedded Programming?

//A note about design patterns in C

// OO or Structured : Better approach OO.

/*File-based  : as CLASS
    =>file can be seen as encapsulating boundary
    => Pair of files  : *.C (src) as funtions body and private varibles/functions
                        *.H (header) as public variables and functions
*/
///////Code : CLASSICAL MODULAR PROGRAMMING : multiple *.c + *.h files ///////

/*Object-based : file base + "structs" to represent the classes (instances of which comprise the
        objects) and 'mangled functions'
        => useful when there will multiple instances
*/

    /*printf(" ::::::::::: TEST I Sensor ::::::::::::: \n");
    printf("Get filterFrequency : %d\n", mySensor.filterFrequency);
    printf("Get updateFrequency : %d\n", mySensor.updateFrequency);
    printf("Get value : %d\n", mySensor.value);

    printf(" ::::::::::: TEST II Sensor ::::::::::::: \n");
    printf("Get filterFrequency : %d\n", Sensor_getFilterFrequency(&mySensor));
    printf("Get updateFrequency : %d\n", Sensor_getUpdateFrequency(&mySensor));
    printf("Get value : %d\n", Sensor_getValue(&mySensor));*/

    /*printf(" ::::::::::: TEST III Sensor ::::::::::::: \n");

    Sensor * p_Sensor0, * p_Sensor1;
    p_Sensor0 = Sensor_Create();
    p_Sensor1 = Sensor_Create();*/

    /* do stuff with the sensors ere */
    /*p_Sensor0->value = 99;
    p_Sensor1->value = -1;
    printf("The current value from Sensor 0 is %d\n", Sensor_getValue(p_Sensor0));
    printf("The current value from Sensor 1 is %d\n", Sensor_getValue(p_Sensor1));*/
    /* done with sensors */
    //Sensor_Destroy(p_Sensor0);
    //Sensor_Destroy(p_Sensor1);

/*Object-oriented : This style is similar to object-based except that the
    => "struct" itself contains "function pointers" (virtual functions)
    => polymorphism and inheritance
*/

// code
    //object instantiation
    /*Sensor * p_Sensor2;
    p_Sensor2 = Sensor_Create();

    //set new frequency
    //Sensor_setFilterFrequency(p_Sensor2, 10);
    //Sensor_getFilterFrequency(p_Sensor2)

    printf(" ::::::::::: TEST I Sensor_OO ::::::::::::: \n");
    printf("Get filterFrequency : %d\n", Sensor_getFilterFrequency(p_Sensor2));
    */
    //printf("Get updateFrequency : %d\n", mySensor.updateFrequency);
    //printf("Get value : %d\n", mySensor.value);
    //Sensor_Destroy(p_Sensor2);


    //:::::::::::::::::::OO second exemple : QUEUE ::::::::::::::::::::::::::::::::::::

    //main_queue();

    //:::::::::::::::: CHACHEDQUEUE exemple : Inheritance / Polymorphism ::::::::::::::

    //main_queue();


//printf("Chapter 2 :  Embedded Programming with The HarmonyTM for Embedded RealTime Process\n");
//Chapter 2  : Embedded Programming with The HarmonyTM for Embedded RealTime Process

//========================================================
//Chapter 3  : Design Patterns for Accessing Hardware
//========================================================
    printf("Chapter 3 :  Design Patterns for Accessing Hardware - TEST\n");

    //HW_main
    main_hw_access();

    printf("//------------------------- End HW Main -------------------------//\n");
//Chapter 4  : Design Patterns for Embedding Concurrency and Resource Management
//Chapter 5 : Design Patterns for State Machines
//Chapter 6  : Safety and Reliability Patterns
//Appendix A UML Notation


    return 0;
}

