#include <stdio.h>
#include <stdlib.h>

/**/
//elements that change regularly should also be *volatile* if being 
//used in an interrupt enabled application 
/* Base, head, tail, should be typed to match the items you add/remove to the buffer*/

typedef struct{
	unsigned char *base;     //Starting point in memory 	//1 byte  
	unsigned char *head;	 //Head pointer 				//1 Byte
	unsigned char *tail;     //Tail Pointer 				//1 Byte
	unsigned int length;	 //Total size of buffer			//4 Bytes
	unsigned int  count;	 //total number of added items   //4 Bytes
}CB_t;
//different ways to implement the buffer
/* /!\ Count can be determined from head, tail and length */
typedef struct{
	unsigned char *base;     //Starting point in memory 	//1 byte  
	unsigned char *head;	 //Head pointer 				//1 Byte
	unsigned char *tail;     //Tail Pointer 				//1 Byte
	unsigned int length;	 //Total size of buffer			//4 Bytes
}CB_t;

typedef enum{
	//CB_ERROR = 0,
	CB_NO_ERROR = 0,
	CB_FULL,
	CB_NOT_FULL,
	CB_EMPLTY,
	CB_NOT_EMPTY,
	CB_NULL,
}CB_status;

 /*
 - Oldest item gets overwritten (NEVER Actually FULL)
 - Head points to next available spot, tail to last added items
 /!\ Prevent overfilling by returning an error
 - track the number of added elements (counts)
 - Use indexes instead of pointers (head, tail)
 */


// Function prototypes

CB_status CB_Is_Buffer_Full(CB_t * cbuff);

//Main 
int main(void)
{

	//Test
	//printf(" circular buffer test\n");
	
	CB_t buf;
	if(buf.length == buf.count)
	{
		return(CB_FULL); // Buffer is Full
	}


	
	return 0; 
}


//functions

//Buffer FULL : Validate fullness using pointers
/* This function could be defined to return a Boolean, full=True, Not full=False*/

CB_status CB_Is_Buffer_Full(CB_t * cbuff)
{
	/* Check if pointers are valid*/
	if( !cbuff || !cbuff->head || !cbuff->tail || !cbuff->base )
	{
		return (CB_NULL);
	}
	if((cbuff->tail == cbuff->head + 1) || (cbuff->head == cbuff-> tail + (cbuff->length - 1)))
	{
		return (CB_FULL);
	}
	else
	{
		return (CB_NOT_FULL);
	}
//Can use multiple if,else if, statement or combine into on conditional

}
//Add Function
CB_status CB_Add_Item(CB_t * cbuff, unsigned char item )
{
	/* Check if pointers are valid*/
	if( !cbuff || !cbuff->head || !cbuff->tail || !cbuff->base )
	{
		return (CB_NULL);
	}
	/* Check if buffer is full */
	if(CB_Is_Buffer_Full(cbuff) == CB_FULL)
	{
		return (CB_FULL);
	}
	
	*cbuf->head = item;
	if((cbuff->head == cbuff-> head + (cbuff->length - 1)))  //Check if wrap aroud or overflow we restart from head : head == base
	{
		cbuff->head = cbuff->base; //Recount from starting point
	}
	else
	{
		cbuf->head++;
	}
	return CB_NO_ERROR;
//Can use multiple if,else if, statement or combine into on conditional

}

//O(1) : No loop


