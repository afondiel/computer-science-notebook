#include <stdio.h>
#include <stdlib.h>

int sommet = 0;

int **Generation_matrice(FILE *p, char *nom_du_fichier){
	int **matrice = NULL;
	int i = 0;
	int j = 0;

	p = fopen(nom_du_fichier, "r");
	fscanf(p, "%d", &sommet);
	
	matrice = (int**)malloc((sommet*sizeof(int*)));

	for(i = 0 ; i < sommet ; i++){
		matrice[i] = (int*)malloc((sommet*sizeof(int)));
	}
	
	for(i = 0 ; i < sommet ; i++){
		for(j = 0 ; j < sommet ; j++){
			fscanf(p, "%d", &matrice[i][j]);
		}
	}
	
	return matrice;
}

void Afficher(int **matrice, int taille, int longeur){
	int i = 0;
	int j = 0;
	
	for(i = 0 ; i < taille ; i++){
		for(j = 0 ; j < longeur ; j++){
			printf("%d ", matrice[i][j]);
		}
	printf("\n");
	}
}

int Nombre_de_poids(int **matrice){
	int i = 0;
	int j = 0;
	int nb_poids = 0;
	
	//Nombre de poids
	for(i = 0; i < sommet ; i++){
		for(j = 0 ; j < sommet ; j++){
			if(matrice[i][j] != 0){
				nb_poids++;
			}
		}
	}
	return nb_poids;
}

int *Generation_du_tableaux_des_pivots(){
	int *tab_pivot = NULL;
	int i = 0;

	tab_pivot = (int*)malloc(sommet*sizeof(int));

	for(i = 0 ; i < sommet ; i++)
		tab_pivot[i] = 0;

	return tab_pivot;
}

int *Generation_du_tableaux_des_paires(){
	int *tab_paire = NULL;

	tab_paire = (int*)malloc(sommet*sizeof(int));

	return tab_paire;
}

int *Generation_du_tableaux_de_Dijkstra(){
	int *tab_dijkstra = NULL;
	int i = 0;

	tab_dijkstra = (int*)malloc(sommet*sizeof(int));

	tab_dijkstra[0] = 0;
	for(i = 1 ; i < sommet ; i++)
		tab_dijkstra[i] = -1;
	
	return tab_dijkstra;
}

int *Tableaux_de_Dijkstra(int **matrice, int *tab_dijkstra, int *tab_pivot){
	int i = 0;	
	int j = 0;
	int k = 0;
	int temp = 0;

	for(i = 0 ; i < sommet ; i ++){
		tab_pivot[temp] = 1;
		for(j = 0 ; j < sommet ; j++){
			if(matrice[temp][j] != 0){
				if((tab_dijkstra[j] > (tab_dijkstra[temp] + matrice[temp][j]))||(tab_dijkstra[j] == (-1))){
					tab_dijkstra[j] = tab_dijkstra[temp] + matrice[temp][j];
				}
			}
		}
		for(k = 1 ; k < (sommet-1) ; k++){
			if(tab_dijkstra[k] > tab_dijkstra[k+1]){
				temp = k;
			}
		}
		for(k = 1 ; k < (sommet-1) ; k++){
			if((tab_dijkstra[temp] > tab_dijkstra[k])&&(tab_dijkstra[k] != (-1))&&(tab_pivot[k] == 0)){
				temp = k;
			}
		}
		for(k = 0 ; k < sommet ; k++){
			printf("%d ", tab_dijkstra[k]);
		}
		printf("\n");
	
	}

	return tab_dijkstra;
}

int main(){
	FILE *p = NULL;
	char *fichier = "Graphique.txt";	
	int **matrice = NULL;
	int **tab_poids = NULL;
	int *tab_pivot = NULL;
	int *tab_dijkstra = NULL;
	int *tab_paire = NULL;
	int *tab_chemin = NULL;
	int poids = 0;
	int i = 0;
	int fin = 0;
	
	//Creation de la matrice
	matrice = Generation_matrice(p , fichier);
	
	printf("Matrice :\n");
	Afficher(matrice, sommet, sommet);

	poids = Nombre_de_poids(matrice);
	printf("Nombre des poids : %d\n", poids);

	tab_pivot = Generation_du_tableaux_des_pivots();
	tab_paire = Generation_du_tableaux_des_paires();
	tab_dijkstra = Generation_du_tableaux_de_Dijkstra();

	printf("Entrer la distance entre 1 et %d :\n", sommet);
	scanf("%d", &fin);

	printf("Tableau de Dijkstra :\n");	
	tab_dijkstra = Tableaux_de_Dijkstra(matrice, tab_dijkstra, tab_pivot);

	printf("Derniere ligne du tableaux :\n");
	for(i = 0 ; i < sommet ; i++){
		printf("%d ", tab_dijkstra[i]);
	}
	printf("\n");

	fin = fin-1;
	printf("La distance est : %d\n", tab_dijkstra[fin]);
	printf("Chemin des sommets choisis :");
	
	tab_chemin = (int*)malloc(sommet*sizeof(int));

	for(i = 0 ; i < sommet ; i++)
		tab_chemin[i] = 0;

	i = 0;
	while(fin != 0){
		tab_chemin[i+1] = tab_pivot[fin];
		fin = tab_pivot[fin]-1;
	}
	for(i = i-1 ; i >=0 ; i --){
		if(i != 0)
			printf("%d", tab_chemin[i]);
		else
			printf("%d\n", tab_chemin[i]);
	}

	return 0;
}
