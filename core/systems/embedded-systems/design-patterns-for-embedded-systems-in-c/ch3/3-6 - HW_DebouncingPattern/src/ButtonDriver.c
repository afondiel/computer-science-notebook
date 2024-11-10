#include "ButtonDriver.h"
#include "Button.h"
#include "MicrowaveEmitter.h"
#include "Timer.h"

/* LOOPS_PER_MS is the # of loops in the delay() function 
	required to hit 1 ms, so it is processor and compiler-dependent
*/
#define LOOPS_PER_MS 1000

void delay(unsigned int ms) {
	long nLoops = ms * LOOPS_PER_MS;
	do{
		while (nLoops–);
	}
}

/* ========= PRIVATE FUNCTIONS =========== */
static void cleanUpRelations(ButtonDriver* const me);

static void cleanUpRelations(ButtonDriver* const me){
	if(me->itsButton != NULL)
	{
		struct ButtonDriver* p_ButtonDriver = Button_getItsButtonDriver(me -> itsButton);
		if(p_ButtonDriver != NULL)
		{
		Button___setItsButtonDriver(me -> itsButton, NULL);
		}
		me->itsButton = NULL;
	}
	if(me->itsMicrowaveEmitter != NULL)
	{
		me->itsMicrowaveEmitter = NULL;
	}

	if(me->itsTimer != NULL)
	{
		me->itsTimer = NULL;
	}
}

/* ============= PUBLIC FUNCTIONS =============== */

void ButtonDriver_Init(ButtonDriver* const me) {
	me->oldState = 0;
	me->toggleOn = 0;
	me->itsButton = NULL;
	me->itsMicrowaveEmitter = NULL;
	me->itsTimer = NULL;
}

void ButtonDriver_Cleanup(ButtonDriver* const me) {
	cleanUpRelations(me);
}

void ButtonDriver_eventReceive(ButtonDriver* const me) {
	//Tempo
	Timer_delay(me->itsTimer, DEBOUNCE_TIME);
	
	if (Button_getState(me->itsButton) != me->oldState) {
		/* must be a valid button event */
		me->oldState = me->itsButton->deviceState;
		if (!me->oldState) {
			/* must be a button release, so update toggle value */
			if (me->toggleOn) {
				me->toggleOn = 0; /* toggle it off */
				Button_backlight(me->itsButton, 0);
				MicrowaveEmitter_stopEmitting(me->itsMicrowaveEmitter);
			}
			else {
				me->toggleOn = 1; /* toggle it on */
				Button_backlight(me->itsButton, 1);
				MicrowaveEmitter_startEmitting(me->itsMicrowaveEmitter);
			}
		}
		/* 
		 * if it’s not a button release, then it must be a button push, which we ignore.
		**/
	}
}

struct Button* ButtonDriver_getItsButton(const ButtonDriver* const me) {
	return (struct Button*)me->itsButton;
}

void ButtonDriver_setItsButton(ButtonDriver* const me, struct Button* p_Button) {
	if(p_Button != NULL)
	{
		Button__setItsButtonDriver(p_Button, me);
	}
	ButtonDriver__setItsButton(me, p_Button);
}

struct MicrowaveEmitter* ButtonDriver_getItsMicrowaveEmitter(const ButtonDriver* const me)
{
	return (struct MicrowaveEmitter*)me->itsMicrowaveEmitter;
}

void ButtonDriver_setItsMicrowaveEmitter(ButtonDriver* const me, struct MicrowaveEmitter* p_MicrowaveEmitter) {
	me->itsMicrowaveEmitter = p_MicrowaveEmitter;
}

struct Timer* ButtonDriver_getItsTimer(const ButtonDriver* const me) {
	return (struct Timer*)me->itsTimer;
}

void ButtonDriver_setItsTimer(ButtonDriver* const me, struct Timer* p_Timer) {
	me->itsTimer = p_Timer;
}

ButtonDriver * ButtonDriver_Create(void) {
	ButtonDriver* me = (ButtonDriver *)
	malloc(sizeof(ButtonDriver));
	if(me != NULL)
	{
		ButtonDriver_Init(me);
	}
	return me;
}

void ButtonDriver_Destroy(ButtonDriver* const me) {
	if(me!=NULL)
	{
		ButtonDriver_Cleanup(me);
	}
	free(me);
}

void ButtonDriver___setItsButton(ButtonDriver* const me, struct Button* p_Button) {
	me->itsButton = p_Button;
}

void ButtonDriver__setItsButton(ButtonDriver* const me, struct Button* p_Button) {
	if(me->itsButton != NULL)
	{
		Button___setItsButtonDriver(me->itsButton, NULL);
	}
	ButtonDriver___setItsButton(me, p_Button);
}

void ButtonDriver__clearItsButton(ButtonDriver* const me) {
	me->itsButton = NULL;
}

