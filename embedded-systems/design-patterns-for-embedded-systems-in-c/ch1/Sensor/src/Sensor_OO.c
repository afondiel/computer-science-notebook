#include <stdio.h>
#include <stdlib.h>
#include "..\include\Sensor_OO.h"

/* Initialize function ptrs in constructor */
void Sensor_Init(Sensor* const me) {
    me->setFilterFreq = Sensor_setFilterFrequency;
    me->getFilterFreq = Sensor_getFilterFrequency;
}

void Sensor_Cleanup(Sensor* const me) {
}

void Sensor_setFilterFrequency(Sensor* const me, int ff){
    me->filterFrequency = ff;
}

int Sensor_getFilterFrequency(const Sensor* const me){
    return me->filterFrequency;
}

/*int Sensor_getUpdateFrequency(const Sensor* const me){
    return me->updateFrequency;
}

void Sensor_setUpdateFrequency(Sensor* const me, int p_updateFrequency) {
    me->updateFrequency = p_updateFrequency;
}

int Sensor_getValue(const Sensor* const me) {
    return me->value;
}
*/
Sensor * Sensor_Create(void) {
    Sensor* me = (Sensor *) malloc(sizeof(Sensor));
    if(me!=NULL)
    {
        Sensor_Init(me);
    }
    return me;
}

void Sensor_Destroy(Sensor* const me) {
    if(me!=NULL)
    {
        Sensor_Cleanup(me);
    }
    free(me);
}
