/*---------------------------------------------------------------------------------
    FILENAME:       main.c

    DESCRIPTION:    Template file containing main()

    DATE:           8 September 2008

    AUTHOR:         W.A. Smith
---------------------------------------------------------------------------------*/

#include "at91sam7s.h"
#include "dbgu.h"

void InitHardware(void);
void TimeDelay(unsigned int ms_delay);

int main(void)
{
    enum machine_state { standby, pump_water, add_flour, mixing };
    enum machine_state state = standby;

    InitHardware();
    while (1) {
        switch (state) {
            case standby:
                /* switch all solenoids off */
                PIO_ODSR = ~0x00;
                /* check for start button pressed */
                if (!(PIO_PDSR & 0x00008000)) {
                    state = pump_water;
                }
            break;
            
            case pump_water:
                /* switch water pump solenoid on for 4s */
                PIO_ODSR = ~0x01;
                TimeDelay(4000);
                state = add_flour;
            break;
            
            case add_flour:
                /* switch flour hopper solenoid on for 2s */
                PIO_ODSR = ~0x02;
                TimeDelay(2000);
                state = mixing;
            break;
            
            case mixing:
                /* mix dough for 5s */
                PIO_ODSR = ~0x04;
                TimeDelay(5000);
                state = standby;
            break;
            
            default:
            break;
        }
    }
}

void InitHardware(void)
{
    /* enable the clock of the PIO controller */
    PMC_PCER = 0x00000004;
    /* configure PIO for switch on PA15 */
    PIO_PER = 0x00008000;
    PIO_ODR = 0x00008000;       /* pins used as input */
    /* configure PIO lines for LEDs on PA0 to PA3 */
    PIO_PER = 0x0000000F;       /* enable uC pins as PIO */
    PIO_OER = 0x0000000F;       /* enable uC pins as output */
    PIO_OWER = 0x0000000F;      /* allow direct writing to port bits */
    PMC_PCER = 0x00001000;      /* enable TC0 clock */
    TC0_CMR = 0x0000C004;       /* set mode, clocksource */
    TC0_RC = 0x0000002F;        /* RC Register value for 1ms */
}

void TimeDelay(unsigned int ms_delay)
{
    TC0_CCR = 0x00000005;       /* start the timer */
    while (ms_delay) {
        if (TC0_SR & 0x00000010)
            ms_delay--;
    }
}
