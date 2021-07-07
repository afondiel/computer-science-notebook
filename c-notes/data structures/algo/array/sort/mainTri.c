#include <stdio.h>
#include <stdlib.h>
#define SIZE 10

/*-------------------------Module principal ---------------------------------*/

int main(void){

	int t[SIZE] = {5,10,4,3,2,1,18,19,15,11};
	
    affichage(t, SIZE);									//array before the sort

	/*triMini(t, SIZE);
	
    affichage(t, SIZE);									//array sorted
	
	triInser(t, SIZE);
	
	affichage(t, SIZE);									//array sorted

	triBulle(t, SIZE);
	
    affichage(t, SIZE); 								//array sorted

	//triShaker(t, SIZE);

	//triFusion(t,SIZE;*/

	triRapide(t, t[0], SIZE);									//quickSort
			
    affichage(t, SIZE); 								//array sorted



	return 0;

}

//keyboardER!!!