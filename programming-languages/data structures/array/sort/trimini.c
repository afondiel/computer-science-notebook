#include <stdio.h>
#include <stdlib.h>

#define SIZE 10


/*Tri par selection de complexité O(n²)*/
//but : ordonner le tableau par ordre croisssant

void tab_display(int *t, int dim);
void trimini(int *t, int n);


int main(void){

	int t[SIZE] = {1,10,2,4,99,5,18,19,15,11};

    tab_display(t, SIZE);
    trimini(t,SIZE);
    tab_display(t, SIZE);




    //getch();
	return 0;

}


void tab_display(int *t, int dim){
	int i;
	for(i = 0; i < dim; i++){
		printf("%d\t", t[i]);
	}
	printf("\n\n");
}


void trimini(int *t, int n){
	int i, j, imin, tmp;
	for(i = 0; i < n; i++){
		imin = i; //imin prend la première valeur de i | indice du plus petit lément de t[n]
		for(j = i+1; j < n; j++){
			if(t[j] < t[imin])
				imin = j;
		}
        tmp = t[imin]; //permutater t[i] avec t[j]
		t[imin] = t[i];
		t[i] = tmp; // tableau trié
	}

}


