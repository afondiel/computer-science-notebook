#include <stdio.h>
#include <stdlib.h>


int main()
{
	/*--------------Déclarations des variables ------------------*/
    FILE *p;
    int *pere = NULL, *pi, *tabPivot = NULL, *sommet, **mat;
    int i,j;
    int MIN, MAX;
    int n = 8;

    /*----------------------------------------------------------------*/
    p=fopen("graphe_o.txt","r");
    fscanf(p,"%d", &n);

    mat=(int**)malloc(n*sizeof(int*));
    for(i=0; i<n; i++)
        mat[i]=(int*)malloc(n*sizeof(int));


    for(i=0; i<n; i++){
        for(j=0; j<n; j++) {
            fscanf(p,"%d", &mat[i][j]);
        }
    }

	/*--------------Affichage de la matrice orientée-----------------*/

	/*for(i=0;i<n;i++){
    	printf("\n");
    	for(j=0;j<n;j++)
    		printf("%d\t", mat[i][j]);

    	printf("\n");
    }
    printf("\n");*/

	/*---------------------------------------------------------------*/


	printf("============ Tableau Pi ============== \n");

	pi=(int*)malloc(n*sizeof(int));

    for(i=0; i<n ;i++){
		pi[i] = -1;
		pi[0] = 0;
		printf("%d | ", pi[i]);
	}

	printf("\n");
	printf("======================================\n\n");


	/*----------------------- Tableau Père----------------------*/

	pere=(int*)malloc(n*sizeof(int)); // Création du tableau père

	/*Remplissage du tableau Pi ????
	 *
	 *
	 * /!\ pivot == sommet
	 * */

	//tabPivot=(int*)malloc(n*sizeof(int)); // Création du tableau des pivots

	int pivot = 0;
	for(i = 0; i < n ; i++)
	{
		for(j = 0; j < n; j++)
		{
			if(mat[pivot][j] != 0)
			{
				if(((pi[pivot]+ mat[pivot][j])< pi[j]) || (pi[j] == -1))
				{
					pi[j] = pi[pivot] + mat[pivot][j];
					pere[j] = pivot;
				}

			}

		}
		//pivot = j;
		//tabPivot[j] = 1;


	}


	/*------------Affichage du Tableau, père et des pivots ---------*/

	printf("--------------Tableau pi -----------\n| ");
	for(i = 0; i<n; i++){
		printf("%d | ", pi[i]);
	}
	printf("\n");
	printf("--------------Tableau père --------------\n| ");

	for(i=0; i<n; i++){
		printf("%d | ", pere[i]);

	}
	printf("\n");
	printf("------------Tableau des pivots --------------\n| ");

	/*for(i=0; i<n; i++){
		printf("%d | ", tabPivot[i]);
	}
	printf("\n\n");*/

	/*-----------------------------------------------------------*/

	/* Recherche du minimum  et du maximum */
	MAX = pi[0];
	MIN = pi[0];
	for (i=0; i<n; i++){
		if((pi[i]< MIN) && (pi[i] > 0) ){
			MIN=pi[i];
		}
		else if(pi[i] > MAX){
			MAX=pi[i];
		}
	}
	printf("Min : %d \n", MIN );
	printf("Max : %d\n", MAX);

	// Création du tableau des sommets " P "
	sommet = (int*)malloc(n*sizeof(int)); 
	
	for(i = 0; i < n; i++)
	{
		sommet[i] = i;
	}
	
	for(i = 0; i < n; i++)
	{
		sommet += tabPivot[i];
	}
	
	for(i = 0; i < n; i++)
    {  
		printf("%d", sommet[i]);
		printf("\n");
        
    }
	

    return 0;

}






