#include <stdio.h>
//Different state of ATM machine
typedef enum
{
    Idle_State,
    Card_Inserted_State,
    Pin_Eentered_State,
    Option_Selected_State,
    Amount_Entered_State,
    last_State
} eSystemState;
//Different type events
typedef enum
{
    Card_Insert_Event,
    Pin_Enter_Event,
    Option_Selection_Event,
    Amount_Enter_Event,
    Amount_Dispatch_Event,
    last_Event
} eSystemEvent;
//typedef of function pointer
typedef eSystemState (*pfEventHandler)(void);
//structure of state and event with event handler
typedef struct
{
    eSystemState eStateMachine;
    eSystemEvent eStateMachineEvent;
    pfEventHandler pfStateMachineEvnentHandler;
} sStateMachine;
//function call to dispatch the amount and return the ideal state
eSystemState AmountDispatchHandler(void)
{
    return Idle_State;
}
//function call to Enter amount and return amount entered state
eSystemState EnterAmountHandler(void)
{
    return Amount_Entered_State;
}
//function call to option select and return the option selected state
eSystemState OptionSelectionHandler(void)
{
    return Option_Selected_State;
}
//function call to enter the pin and return pin entered state
eSystemState EnterPinHandler(void)
{
    return Pin_Eentered_State;
}
//function call to processing track data and return card inserted state
eSystemState InsertCardHandler(void)
{
    return Card_Inserted_State;
}
//Initialize array of structure with states and event with proper handler
sStateMachine asStateMachine [] =
{
    {Idle_State,Card_Insert_Event,InsertCardHandler},
    {Card_Inserted_State,Pin_Enter_Event,EnterPinHandler},
    {Pin_Eentered_State,Option_Selection_Event,OptionSelectionHandler},
    {Option_Selected_State,Amount_Enter_Event,EnterAmountHandler},
    {Amount_Entered_State,Amount_Dispatch_Event,AmountDispatchHandler}
};
//main function
int main(int argc, char *argv[])
{
    eSystemState eNextState = Idle_State;
    while(1)
    {
        //Api read the event
        eSystemEvent eNewEvent = read_event();
        if((eNextState < last_State) && (eNewEvent < last_Event)&& (asStateMachine[eNextState].eStateMachineEvent == eNewEvent) && (asStateMachine[eNextState].pfStateMachineEvnentHandler != NULL))
        {
            // function call as per the state and event and return the next state of the finite state machine
            eNextState = (*asStateMachine[eNextState].pfStateMachineEvnentHandler)();
        }
        else
        {
            //Invalid
        }
    }
    return 0;
}

//https://docs.staruml.io/working-with-uml-diagrams/statechart-diagram