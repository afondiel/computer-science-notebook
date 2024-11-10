/*C Object Oriented Design (header file)*/
//The function pointers support polymorphism and virtual functions

#ifndef proxy_pattern_H
#define proxy_pattern_H


struct MotorController;
struct MotorData;
struct MotorDisplay;
struct MotorProxy;

typedef enum DirectionType {
    NO_DIRECTION,
    FORWARD,
    REVERSE
} DirectionType;

#endif /*proxy_pattern_H*/
