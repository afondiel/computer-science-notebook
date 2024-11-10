#include <stdio.h>
//Different state of ATM machine
typedef enum
{
    Idle_State,
    Card_Inserted_State,
    Pin_Eentered_State,
    Option_Selected_State,
    Amount_Entered_State,
} eSystemState;
//Different type events
typedef enum
{
    Card_Insert_Event,
    Pin_Enter_Event,
    Option_Selection_Event,
    Amount_Enter_Event,
    Amount_Dispatch_Event
} eSystemEvent;
//Prototype of eventhandlers
eSystemState AmountDispatchHandler(void)
{
    return Idle_State;
}
eSystemState EnterAmountHandler(void)
{
    return Amount_Entered_State;
}
eSystemState OptionSelectionHandler(void)
{
    return Option_Selected_State;
}
eSystemState EnterPinHandler(void)
{
    return Pin_Eentered_State;
}
eSystemState InsertCardHandler(void)
{
    return Card_Inserted_State;
}
int main(int argc, char *argv[])
{
    eSystemState eNextState = Idle_State;
    eSystemEvent eNewEvent;
    while(1)
    {
        //Read system Events
        eSystemEvent eNewEvent = ReadEvent();
        switch(eNextState)
        {
        case Idle_State:
        {
            if(Card_Insert_Event == eNewEvent)
            {
                eNextState = InsertCardHandler();
            }
        }
        break;
        case Card_Inserted_State:
        {
            if(Pin_Enter_Event == eNewEvent)
            {
                eNextState = EnterPinHandler();
            }
        }
        break;
        case Pin_Eentered_State:
        {
            if(Option_Selection_Event == eNewEvent)
            {
                eNextState = OptionSelectionHandler();
            }
        }
        break;
        case Option_Selected_State:
        {
            if(Amount_Enter_Event == eNewEvent)
            {
                eNextState = EnterAmountHandler();
            }
        }
        break;
        case Amount_Entered_State:
        {
            if(Amount_Dispatch_Event == eNewEvent)
            {
                eNextState = AmountDispatchHandler();
            }
        }
        break;
        default:
            break;
        }
    }
    return 0;
}

//https://aticleworld.com/state-machine-using-c/

//https://docs.staruml.io/working-with-uml-diagrams/statechart-diagram

