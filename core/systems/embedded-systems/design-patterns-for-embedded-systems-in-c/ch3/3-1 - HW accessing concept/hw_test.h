#ifndef hw_test_h
#define hw_test_h

// #83 : Manipulating bit-oriented memory mapped hardware
#define TURN_OFF (0x00)
#define INITIALIZE (0x61)
#define RUN (0x69)
#define CHECK_ERROR (0x02)
#define DEVICE_ADDRESS (0x01FFAFD0)

// Functions prototypes

extern void emergencyShutDown(void);
extern void bitField_TEST1(void);
extern void bitField_TEST2(void);


#endif /* hw_test_h */
