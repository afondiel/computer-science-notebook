#include <stdio.h>
#include <stdlib.h>


typedef struct{
	unsigned char *base;     //Starting point in memory 	//1 byte  
	unsigned char *head;	 //Head pointer 				//1 Byte
	unsigned char *tail;     //Tail Pointer 				//1 Byte
	unsigned int length;	 //Total size of buffer			//4 Bytes
	unsigned int  count;	 //total number of added items   //4 Bytes
}LLB_t;

int main(void)
{



	return 0;

}

/* ==================== NOTES =======================

 /!\ Not contigous memory 
 - LL can be allocated in random location of the memory (Not linear)





*/
