#include <stdio.h>
#include <stdlib.h>
#define SIZE 10

/*Complexité en espace : En raison des appels récursif, on a besoin d'une pile dont la taille est en O(log(n)).
Complexité en temps :  O(n.log(n)) en moyenne,  O(n^{2}) dans le pire cas.
Nom anglais : quicksort.*/

void tab_display(int *t, int dim){

void trimini(int *t, int n);

int main(void){

	int t[10] = {1,10,2,4,99,5,18,19,15,11};
	int i, n = 10;
	//textcolor(3);
	for(i = 0; i<n; i++)
        printf("%d\t", t[i]);
    printf("\n\n");
    trimini(t, n);
    printf("\n\n");

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
	int i, j, imin, aux;
	for(i = 0; i < n; i++){
		imin = i; //imin prend la première valeur de i | indice du plus élément de t[n]
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

//quickSort

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
