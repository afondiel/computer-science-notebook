///////////////////////////////////////////////////////
// design-patterns-for-embedded-systems-in-c
// by : Bruce Powel Douglass - PhD
///////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "ch1\Sensor\include\Sensor.h"

Sensor mySensor = {
                   //Hz  //Hz //mV
                    100, 60, 5000
                  };

///////////////////// MAIN TEST //////////////////////////
int main()
{


    // CONTENTS

    printf("Chapter 1 :  What Is Embedded Programming?\n");

//Chapter 1 :  What Is Embedded Programming?

//A note about design patterns in C

// OO or Structured : Better approach OO.

/*File-based */

//file can be seen as encapsulating boundary*/
//      => Pair of files  : *.C (src) as funtions body and private varibles/functions
//                          *.H (header) as public variables and functions
    printf(" ::::::::::: TEST Sensor ::::::::::::: \n");
    printf("Get filterFrequency : %d\n", mySensor.filterFrequency);
    printf("Get updateFrequency : %d\n", mySensor.updateFrequency);
    printf("Get value : %d\n", mySensor.value);


/*Object-based*/

/*Object-oriented*/

//printf("Chapter 2 :  Embedded Programming with The HarmonyTM for Embedded RealTime Process\n");
//Chapter 2  : Embedded Programming with The HarmonyTM for Embedded RealTime Process

//printf("Chapter 3 :  Design Patterns for Accessing Hardware\n");
//Chapter 3  : Design Patterns for Accessing Hardware
//Chapter 4  : Design Patterns for Embedding Concurrency and Resource Management
//Chapter 5 : Design Patterns for State Machines
//Chapter 6  : Safety and Reliability Patterns
//Appendix A UML Notation




    return 0;
}
