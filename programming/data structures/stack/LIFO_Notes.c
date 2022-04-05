#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
