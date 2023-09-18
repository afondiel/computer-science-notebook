#include <stdio.h>
#include <stdlib.h>
#define SIZE 10


//Tri à bulle de complexité O(n²)*/
//but : ordonner le tableau par ordre croisssant

void tab_display(int *t, int dim);
void triBulle(int *t, int n);

int main(void){

	int t[SIZE] = {5,10,2,4,7,3,18,19,15,11};

	tab_display(t, SIZE);
    triBulle(t,SIZE);
    tab_display(t, SIZE);


	return 0;

}

void tab_display(int *t, int dim){
	int i;
	for(i = 0; i < dim; i++){
		printf("%d\t", t[i]);
	}
	printf("\n\n");
}

void triBulle(int *t, int n){
	int i, j, aux;
	for(i = 0; i < n; i++){
		for(j = n-1; j > i; j--){
			if(t[j-1] > t[j]){
                aux = t[j];
                t[j] = t[j-1];
                t[j-1] = aux;
			}
		}
	}

}

