#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "functions.h"

int main(){

	int i;
	float tab2D[SIZE];
	float tab1D[SIZE];
	float w[2]={1,2};
	for(i=0;i<2;i++)
		printf("%f\n",w[i]);

	TAB_init(tab1D, SIZE);
	TAB_fill_increase(tab1D, SIZE);
	TAB_display(tab1D, SIZE);
	TAB_fill_random(tab2D, SIZE);
	TAB_display(tab2D, SIZE);
	/*TAB_copy(tab2D,tab1D,SIZE);
	TAB_display(tab2D, SIZE);*/
	/*TAB_switch(tab2D, SIZE, 1, 4);*/
	/*printf("\nThe sum is %f\n",TAB_sum(tab2D, SIZE));
	printf("\nThe mean is %f\n",TAB_mean(tab2D, SIZE));
	printf("\nThe min is %f\n",TAB_min(tab2D, SIZE));
    printf("\nThe Index of min is %d and the result is %f\n",TAB_index_min_range(tab2D,SIZE,3,6),tab2D[TAB_index_min_range(tab2D,SIZE,3,6)]);
	*/
	/*TAB_sort_asc(tab2D, SIZE);
	TAB_display(tab2D, SIZE);*/




	return EXIT_SUCCESS;
}
