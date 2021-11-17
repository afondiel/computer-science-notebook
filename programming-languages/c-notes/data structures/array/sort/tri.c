#include "tri.h"

/*-------------------------Module secondaire ---------------------------------*/


void affichage(int *t, int n){
	int i;
	for(i = 0; i < n; i++){
		printf("%d\t", t[i]);
	}
	printf("\n\n");
}

//TRI SELECTION - de recherche du MINImum

/*Tri par selection de complexité O(n²)
but : ordonner le tableau par ordre croisssant */

void triMini(int *t, int n){
	int i, j, imin, aux;
	for(i = 0; i < n; i++){
		imin = i; //imin prend la première valeur de i | indice du plus petit élément de t[n]
		for(j = i+1; j < n; j++){
			if(t[j] < t[imin])
				imin = j;
		}
        aux     = t[imin]; //permutater t[i] avec t[j]
		t[imin] = t[i];
		t[i]    = aux; // tableau trié
	}

}
//TRI INSERTION

//Tri par insertion de complexité O(n²)*/
//but : ordonner le tableau par ordre croisssant

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
        for(j=0; j < n ; j++){
            if(t[j] < t[j-1]){
                aux    = t[j];
                t[j]   = t[j-1];
                t[j-1] = aux;
            }

        }
	}
}

//TRI BULLE
//Tri à bulle de complexité O(n²)*/
//but : ordonner le tableau par ordre croisssant

void triBulle(int *t, int n){
	int i, j, aux;
	for(i = 0; i < n; i++){
		for(j = n-1; j > i; j--){
			if(t[j] < t[j-1]){
                aux    = t[j];
                t[j]   = t[j-1];
                t[j-1] = aux;
			}
		}
	}

}

//TRI SHAKER
void triShaker(int *t, int n){
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

 //TRI FUSION -------- A modifier ?????????
void triFusion(int *t, int n){
	int i, j, aux,imin;
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

//TRI RAPIDE == quickSort


/*Complexité en espace : En raison des appels récursif, on a besoin d'une pile dont la taille est en O(log(n)).
Complexité en temps :  O(n.log(n)) en moyenne,  O(n^{2}) dans le pire cas.
Nom anglais : quicksort.*/

/*FONCTION Partitionner ( A : tableau [1..n] d’Entiers, p : Entier, r : Entier) : Entier
x, i, j, temp: Entier
bool: Booleen
Début
  x :=     A[p]
  i :=     p-1
  j :=     r+1
  bool := vrai
  Tant que (bool) Faire
    Répéter j := j-1 Jusqu'à A[j] <= x
    Répéter i := i+1 Jusqu'à A[i] >= x
    bool := faux
    Si  i < j
      temp := A[i]
      A[i] := A[j]
      A[j] := temp
      bool := vrai
    Sinon
        Retourner j
    Fin si
  Fin tant que
Fin

PROCÉDURE Tri_rapide(A : tableau [1..n], p : entier, r : entier)
q : Entier
Début
  Si  p < r
    q := Partitionner(A,p,r)
    Tri_rapide(A,p,q)
    Tri_rapide(A,q+1,r)
  Fsi
Fin
*/

//!\ Par recursivité

/*int partition(int *t, int p, int r){
	int x, i, j, aux;
	x = t[p];
	i = p-1;
	j = r+1;
	//bool = vrai;

	while(bool){
		for(j = j-1; t[j] <= x;j++)
			fot(i = i+1 ; t[i] >= x; i++){
			//bool = faux;
			if(i < j){
				aux  = t[i];
				t[i] = t[j];
				t[j] = aux;
				//bool = vrai;
			}
			else
				return j;		
			}
		}
	}
	
}

void triRapide(int *t, int p , int r){			//quickSort

	int q;
	q = partition(*t,p,r);
	
	if( p < r){
		 triRapide(*t, p, q);
		 triRapide(*t, 	q+1, r);
	}

}*/

//Inspired in the document tri.pdf & algo_de_tri.pdf "C:\Users\Afonso Diela\Desktop\ITI\Info\Tri"

int partition(int *t, int g, int d){
	/*g=gauche ; d = droite */
	int i,j, pivot, aux1, aux2;
	pivot = t[g];
	i     = g + 1; 			//indice où inserer le 1er élément <=t[g]
	for(j = g + 1; j < d; j++){
		/* j est l'indice de l'élément courant */
		if(t[j] <= pivot){
			aux1 = t[i];
			t[i] = t[j];
			t[j] = aux1;
			i    = i + 1;
		}
	}
	aux2      = t[g];
	t[g]      = t[i - 1];
	t[i - 1]  = aux2;
	
	return (i - 1);
}

void triRapide(int *t, int g , int d){			//quickSort

	int indPivot;
	
	if( g < d){
		indPivot = partition(t, g, d);
		triRapide(t, g, indPivot - 1);
		triRapide(t, indPivot + 1, d);
	}

}

