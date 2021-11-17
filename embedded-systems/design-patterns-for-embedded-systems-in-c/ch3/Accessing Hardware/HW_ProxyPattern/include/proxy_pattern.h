/*C Object Oriented Design (header file)*/
//The function pointers support polymorphism and virtual functions

#ifndef proxy_pattern_H
#define proxy_pattern_H

typedef enum{
	MEMORYMAPPED,
	PORTMAPPED
}e_interface;

/* function pointers */
typedef int (*f0ptrInt) (const Sensor* const me);        /* (void*) == ptr to the function w only me ptr argument */
typedef void (*f1ptrVoid) (const Sensor* const me, int); /* (void*, int)  == ptr to function with me ptr + int args */

/* class Sensor */
typedef struct {
    int filterFrequency;
    int updateFrequency;
    int value;
    //ADConverter* myADConvert; /* association implemented as ptr */

    // virtual functions
    f0ptrInt getFilterFreq; /* ptr to the function w only me ptr argument */
    f1ptrVoid setFilterFreq; /* ptr to function with me ptr and int args */
    e_interface whatKindOfInterface; /*sensor interface cmd : memory/port mapped */

}Sensor;


int Sensor_getFilterFrequency (const Sensor* const me);
void Sensor_setFilterFrequency(const Sensor* const me, int ff);
Sensor * Sensor_Create (void);              /* creates struct and calls init */
void Sensor_Init(Sensor* const me);         /* initializes vars incl. function ptrs */
void Sensor_Destroy(Sensor* const me);



#endif /*proxy_pattern_H*/
