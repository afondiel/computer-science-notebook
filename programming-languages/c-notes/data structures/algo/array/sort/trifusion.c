#include <stdio.h>
#include <stdlib.h>
#define SIZE 10

/*Tri par insertion de complexité O(n*log(n))

Complexité en temps : O(n.log(n))
Nom anglais : merge sort.*/

void tab_display(int *t, int dim);
void trifusion(int *t, int n);

int main(void){

	int t[10] = {1,10,2,4,99,5,18,19,15,11};

    tab_display(t, SIZE);
    trifusion(t, SIZE);
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
void trifusion(int *t, int n){
	int i, j, aux;
	for(i = 0; i < n; i++){
		imin = i; //imin prend la première valeur de i
		for(j = i+1; j < n; j++){
			if(t[j] < t[imin])
				imin = j;
		}
        aux = t[imin];
		t[imin] = t[i];
		t[i] = aux;
        printf("%d\t", t[i]);
	}

}

/*#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
void fusion(int tableau[],int deb1,int fin1,int fin2)
        {
        int *table1;
        int deb2=fin1+1;
        int compt1=deb1;
        int compt2=deb2;
        int i;

        table1=malloc((fin1-deb1+1)*sizeof(int));

        //on recopie les éléments du début du tableau
        for(i=deb1;i<=fin1;i++)
            {
            table1[i-deb1]=tableau[i];
            }

        for(i=deb1;i<=fin2;i++)
            {
            if (compt1==deb2) //c'est que tous les éléments du premier tableau ont été utilisés
                {
                break; //tous les éléments ont donc été classés
                }
            else if (compt2==(fin2+1)) //c'est que tous les éléments du second tableau ont été utilisés
                {
                tableau[i]=table1[compt1-deb1]; //on ajoute les éléments restants du premier tableau
                compt1++;
                }
            else if (table1[compt1-deb1]<tableau[compt2])
                {
                tableau[i]=table1[compt1-deb1]; //on ajoute un élément du premier tableau
                compt1++;
                }
            else
                {
                tableau[i]=tableau[compt2]; //on ajoute un élément du second tableau
                compt2++;
                }
            }
        free(table1);
        }


void tri_fusion_bis(int tableau[],int deb,int fin)
        {
        if (deb!=fin)
            {
            int milieu=(fin+deb)/2;
            tri_fusion_bis(tableau,deb,milieu);
            tri_fusion_bis(tableau,milieu+1,fin);
            fusion(tableau,deb,milieu,fin);
            }
        }

void tri_fusion(int tableau[],int longueur)
     {
     if (longueur>0)
            {
            tri_fusion_bis(tableau,0,longueur-1);
            }
     }


*/
