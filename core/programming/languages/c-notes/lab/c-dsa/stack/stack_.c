#include <stdio.h>
#include <stdlib.h>

/*
    data structure based on  : LIFO = LAST IN FIRST OUT
    for accessing and processing the data
*/

/*
|   last   |
----------
|          |
------------
|          |
----------
|   first  |
------------
*/

/*---- structure-----*/
//=== declaration ====
//stack<type> _stack
//=== definition ======
//stack<type> _stack = {1, 2, 3};

// functions ----------------------x------------------------------ Time complexity
/*
- empty() :  returns whether the stack is empty                     ------------- O(1)
- size() : returns the size of the stack                            ------------- O(1)
- top() : returns a reference to the top most element of the stack  ------------- O(1)
- push(g) : adds the element 'g' at the top of the stack            ------------- O(1)
- pop() : deletes the top most element of the stack                 ------------- O(1)
-
*/

/*

====== = = =  LIFO (Last In First Out) = = =  = ===========

DATA structures help >organize< embedded system control and >storage< of data.

LIFO Buffer :

-------------------------------------
Producer => Data Buffer ==> Consumer
-------------------------------------

Producer : Creates and packages data from sensors or connected devices

Data Buffer :  data structure in shared memory implemended between two processing interfaces.

Consumer : Processes data to make decisions, control outputs or communicate externally

-----------
COMMANDS :
-----------

	-> PUSH : to add data INTO the buffer
	-> POP : to get out data FROM the buffer

LIFO : Adds and removes data items to/from the same end (created with a contigous block of memory like *LINKED LIST*)

=> referred to as a STACK
=> Similar to processor stack, but diferent

*/

/*------------------------------------------------------------------
//LIFO DATA STRUCTURE Definitions
-----------------------------------------------------------------*/
//Implementation 1
typedef struct{
	unsigned int length;    //total size of the buffer
	unsigned char * base;	//Starting point in the memory
	unsigned int counter;	//Total number of added items
}LIFO_Buf_t1;

//Implementation 2
typedef struct{
	unsigned int length;    //total size of the buffe
	unsigned char * base;   //Starting point in the memory
	unsigned char * head;	//Most recent added items
}LIFO_Buf_t2;

typedef enum {
	LB_ERROR=0,
	LB_NO_ERROR,
	LB_FULL,
	LB_NOT_FULL,
	LB_EMPTY,
	LB_NOT_EMPTY,
	LB_NULL,
}LB_Status_e;

/*-------------------------------------------------------------------
//functions prototypes
	- constructor
	- destructor
	- Modifier
	- Iterator
---------------------------------------------------------------------*/
//Buffer_push();     //Add item into the buffer
//Buffer_pop();	    //get items from the buffer
//Buffer_status();  //Displays  the status of the LIFO


//--------------------------------------------------------------------

//functions definitions

//Buffer Full

LB_Status_e LIFO_Is_Buf_Full(LIFO_Buf_t2 *lbuf2)
{
	/*Check if pointer are valid*/
	if(!lbuf2||!lbuf2->base||!lbuf2->head)
	{
		return (LB_NULL);
	}
	if(lbuf2->head >= (lbuf2->base + lbuf2->length)) //Address operations
	{
		return(LB_FULL);
	}
	else
	{
		return (LB_NOT_FULL);
	}
}

//Buffer Add

LB_Status_e LIFO_Add_Item(LIFO_Buf_t2 *lbuf2, unsigned char item)
{
	/*Check if pointers are valid*/
	if(!lbuf2||!lbuf2->base||!lbuf2->head)
	{
		return (LB_NULL);
	}
	/*check if buffer is full*/
	if((LIFO_Is_Buf_Full(lbuf2) == LB_FULL)) //To prevent Buffer Overflow!
	{
		return(LB_FULL);
	}
	*lbuf2->head = item;
	lbuf2->head++;
	return LB_NO_ERROR;
}



//Interfaces


//-------------------------------------------------------------------
int main(void)
{
	unsigned int length;


	printf("$$ TEST1 $$\n ");

	//In this machine : ASUS x64
	/*printf(" char : %d%\n ",sizeof(char));     //1 byte
	printf(" int : %d%\n ",sizeof(int));	   //4 bytes
	printf(" long : %d%\n ",sizeof(long));	   //4 bytes
	printf(" float : %d%\n ",sizeof(float));   //4 bytes
	printf(" double : %d%\n ",sizeof(double)); //8 bytes */


	//	>>>> SRAM <<<<<
	/* 	| Buffer(lenght)|
		-----------------
base ->		items(n)
		-----------------
		-----------------
		-----------------
				.
				.
				.
		-----------------
head->
		-----------------*/

	//LIFO Implementation 2
	LIFO_Buf_t2 lbuf2; 								//create instance
	LB_Status_e lb_stat;
	lbuf2.base = (unsigned char *)malloc(sizeof(length));  //dynamic method

	//unsiged char buffer[length]; 	//array
	//lbuf2.base = buffer;			//static method :: without a malloc which SAFER for embedded systems!!!!

	if(!lbuf2.base)
	{
		return (LB_ERROR);
	}
	lbuf2.length = length;
	//At the beguinning head is in the same position as base
	//It increases with "length"
	lbuf2.head =  lbuf2.base;  //Head has the ADDRESS same size as base

	//>>>>>>>>>>>>>>>>>>>> TEST <<<<<<<<<<<<<<<<<<<<<

	//Add Items ?

	//*display LIFO  ?

	 return 0;
}
