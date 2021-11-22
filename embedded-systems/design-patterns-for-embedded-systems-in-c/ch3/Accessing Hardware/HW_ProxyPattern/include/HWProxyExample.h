/*C Object Oriented Design (header file)*/
//The function pointers support polymorphism and virtual functions

#ifndef HWProxyExample_H
#define HWProxyExample_H

//HW Proxy Example **INTERFACE**

// client proxy
struct MotorController;
struct MotorDisplay;
// HW data
struct MotorData;

//HW
struct MotorProxy;

typedef enum DirectionType {
    NO_DIRECTION,
    FORWARD,
    REVERSE
} DirectionType;

#endif /*HWProxyExample_H*/
