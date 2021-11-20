//#include <stdio.h>
//#include <stdlib.h>
//#include "..\include\proxy_pattern.h"
//
///* Initialize function ptrs in constructor */
//void Sensor_Init(Sensor* const me) {
//    me->setFilterFreq = Sensor_setFilterFrequency;
//    me->getFilterFreq = Sensor_getFilterFrequency;
//}
//
//void Sensor_Cleanup(Sensor* const me) {
//}
//
//void Sensor_setFilterFrequency(Sensor* const me, int ff){
//    me->filterFrequency = ff;
//}
//
//int Sensor_getFilterFrequency(const Sensor* const me){
//    return me->filterFrequency;
//}
//
///*int Sensor_getUpdateFrequency(const Sensor* const me){
//    return me->updateFrequency;
//}
//
//void Sensor_setUpdateFrequency(Sensor* const me, int p_updateFrequency) {
//    me->updateFrequency = p_updateFrequency;
//}
//
//int Sensor_getValue(const Sensor* const me) {
//    return me->value;
//}
//*/
//Sensor * Sensor_Create(void) {
//    Sensor* me = (Sensor *) malloc(sizeof(Sensor));
//    if(me!=NULL)
//    {
//        Sensor_Init(me);
//    }
//    return me;
//}
//
//void Sensor_Destroy(Sensor* const me) {
//    if(me!=NULL)
//    {
//        Sensor_Cleanup(me);
//    }
//    free(me);
//}
//
//
//// Polymorphism in the hard way
//int acquireValue(Sensor *me) {
//    int *r, *w; /* read and write addresses */
//    int j;
//    switch(me->whatKindOfInterface) {
//        case MEMORYMAPPED:
//            w = (int*)WRITEADDR; /* address to write to sensor */
//            *w = WRITEMASK; /* sensor command to force a read */
//            for (j=0;j<100;j++) { /* wait loop */ };
//            r = (int *)READADDR; /* address of returned value */
//            me->value = *r;
//        break;
//        case PORTMAPPED:
//            me->value = inp(SENSORPORT);
//            /* inp() is a compiler-specific port function */
//        break;
//    }; /* end switch */
//    return me->value;
//};
