#include <stdio.h>
#include <stdlib.h>

int sommet = 0;
int nb_poids = 0;

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

int **Tableau_de_poids(int **matrice){
	int i = 0;
	int j = 0;
	int col = 0;
	int lig = 0;
	int **tab_poids = NULL;
	
	//Nombre de poids
	for(i = 0; i < sommet ; i++){
		for(j = 0 ; j < sommet ; j++){
			if(matrice[i][j] != 0){
				nb_poids++;
			}
		}
	}
	printf("Nombre des poids : %d\n", nb_poids);

	//Generation du tableau des poids
	tab_poids = (int**)malloc((nb_poids*sizeof(int*)));
	for(i = 0 ; i < nb_poids ; i ++)
		tab_poids[i] = (int*)malloc(3*sizeof(int));
	
	for(i = 0 ; i < sommet ; i++){
		for(j = 0 ; j < sommet ; j++){
			if(matrice[i][j] != 0){
				tab_poids[col][lig] = i+1;
				tab_poids[col][lig+1] = j+1;
				tab_poids[col][lig+2] = matrice[i][j];
				col++;
			}
		}
	}

	return tab_poids;
}

int main(){
	FILE *p = NULL;
	int **matrice = NULL;
	int **tab_poids = NULL;
	char *fichier = "Graphique.txt";
	int *tab_pivot = NULL;
	int *tab_dijkstra = NULL;
	int *tab_paire = NULL;
	int pivot = 0;
	int i_temp = 0;
	int temp = 999;
	int i = 0;
	int j = 0;
	int k = 0;
	
	//Creation de la matrice
	matrice = Generation_matrice(p , fichier);
	
	printf("Matrice :\n");
	Afficher(matrice, sommet, sommet);
	
	tab_poids = Tableau_de_poids(matrice);
	printf("Tableau des poids :\n");	
	Afficher(tab_poids, nb_poids, 3);

	printf("Tableau des pivots :\n");
	tab_pivot = (int*)malloc(sommet*sizeof(int));
	tab_paire = (int*)malloc(sommet*sizeof(int));
	tab_dijkstra = (int*)malloc(sommet*sizeof(int));

	for(i = 0 ; i < sommet ; i++){
		tab_pivot[i] = i+1;
		tab_dijkstra[i+1] = 9999;
		printf("%d ", tab_pivot[i]);
	}
	printf("\n");

	tab_dijkstra[0] = 0;
	i = 0;

	printf("Tableau de Dijkstra :\n");
	while(i < sommet){
		if(tab_pivot[i] > 0){
			printf("Se place sur le pivot \n");
			tab_paire[i] = tab_pivot[i];
			pivot = tab_pivot[i];
			tab_pivot[i] = 0;
			for(j = 0 ; j < nb_poids ; j++){
				printf("%d / %d \n", tab_poids[j][0], pivot);
				if(tab_poids[j][0] == pivot){
					printf("Pivot trouvé \n");
					i_temp = tab_poids[j][1];					
					if(tab_poids[j][2] < tab_dijkstra[(i_temp-1)]){
						printf("Mise à jour tableau de dijkstra \n");
						tab_dijkstra[(i_temp-1)] = tab_poids[j][2];
						if((temp > tab_poids[j][2])&&(tab_pivot[(i_temp-1)] != 0)){
							i = tab_poids[j][1];
							temp = tab_poids[j][2];
						}
					}
				}
			}
			printf("prochain pivot : %d\n", i);
		}
	}
	
	for(i = 0 ; i < sommet ; i++){
		printf("%d ", tab_dijkstra[i]);
	}
	printf("\n");

	return 0;
}
