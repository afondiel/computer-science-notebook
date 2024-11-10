#include <stdio.h>
#include <stdlib.h>
#define SIZE 10
//Tri par insertion de complexité O(n²)*/
//but : ordonner le tableau par ordre croisssant

void tab_display(int *t, int dim);
void triInser(int *t, int n);

int main(void){

	int t[SIZE] = {5,10,4,3,2,1,18,19,15,11};

    tab_display(t, SIZE);
    triInser(t, SIZE);
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

/******Il traite pas la prémière valeur !!! **************************/

/*void triInser(int *t, int n){
	int i, j,aux;
	for(i = 2; i <= n; i++){
        aux = t[i];
        j=i;
        while((j>1)&&(aux<t[j-1])){
            t[j] = t[j-1]; // on décale les éléments pas triées et on insère un nouveau élément
            j--;
        }
        t[j] = aux; //Insertion
	}

}*/
void triInser(int *t, int n){
	int i, j,aux;
	for(i = 1; i < n; i++){
        for(j=i; j>0; j--){
            if(t[j]<t[j-1]){
                aux = t[j];
                t[j] = t[j-1];
                t[j-1] = aux;
            }

        }
	}
}

