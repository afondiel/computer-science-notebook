#include <stdio.h>
#include <stdlib.h>
#define SIZE 10

//Tri shaker de complexité O(n²)*/
//but : ordonner le tableau par ordre croisssant

void tab_display(int *t, int dim);
void trishaker(int *t, int n);


int main(void){

	int t[SIZE] = {5,10,2,4,7,5,18,1,15,11};

	tab_display(t, SIZE);
    trishaker(t, SIZE);
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

void trishaker(int *t, int n){
	int i, j=1,aux, g=0,d=n-1;
	while(g<d){
        for(i = g; i < d; i++){
            if(t[i] > t[i+1]){
                aux = t[i];
                t[i] = t[i+1];
                t[i+1] = aux;
                j=i;
            }
            d=j;
            printf("%d\t", d);
        }
        for(j = i+1; j < n; j++){
			if(t[i] < t[i-1]){
                aux = t[i];
                t[i] = t[i-1];
                t[i-1] = aux;
                j=i;
			}
            g=j;
        }
        printf("%d\t",g);
        i++;

	}

}


//Cocktail sort

//algo :
/*fonction tri_cocktail (array liste)
    échangé  := vrai
    Répéter tant que échangé = vrai
        échangé := faux

         Répéter pour tout  i entre 0 et liste.taille - 2
            si liste[i] > liste[i + 1]
                [[Echanger (liste[i], liste[i+1])
                échangé  := vrai
            fin si
         fin Répéter
         Répéter pour tout  i (décroissant) entre liste.taille-2 et 0
            si liste[i] > liste[i + 1]
                [[Echanger (liste[i], liste[i+1])
                échangé  := vrai
            fin si
         fin Répéter
     fin tant que
 fin fonction

 */

