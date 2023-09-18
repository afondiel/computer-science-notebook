#include <stdio.h>
#include <stdlib.h>




int main()
{

    /*--------------Déclarations des variables locales ------------------*/
    FILE *p;
    int *etiq;
    int **t_aretes, **mat;
    int i,j,k,l,aux1,aux2,aux3,n=6;
    /*-------------------------------------------------------------------*/
    p=fopen("graph.txt","r");
    fscanf(p,"%d", &n);

    mat=(int**)malloc(n*sizeof(int*));
    for(i=0; i<n; i++)
        mat[i]=(int*)malloc(n*sizeof(int));
        for(i=0; i<n; i++)
    {
        for(j=0; j<n; j++)
        {
            fscanf(p,"%d", &mat[i][j]);

        }
    }
    /*--------------Affichage de la matrice--------------------*/
    /*for(k=0;k<n;k++){
    	printf("\n");
    	for(l=0;l<n;l++)
    		printf("%d\t", mat[k][l]);

    	printf("\n");
    }
    printf("\n");*/
	/*--------------Affichage de la demie matrice----------------*/
    /*for(k=0;k<n;k++){ //To print the half of the matrix
    	printf("\n");
    	for(l=k+1;l<n;l++)
    		printf("%d\t", mat[k][l]);
    		printf("\n");
    }*/
	/*------------------------------------------------------------*/

    int nbar=0;

    for(i=0; i<n; i++)
    {
        for(j=i; j<n; j++)
            if(mat[i][j]!=0)
                nbar++;
    }
    //printf("%d\n", nbar);


    t_aretes=(int**)malloc(nbar*sizeof(int*)); // on alloue de la mémoire pour le t_aretes
    for(i=0; i<nbar; i++)
        t_aretes[i]=(int*)malloc(3*sizeof(int));
    k=0;
    for(i=0; i<n; i++)
    {
        for(j=i+1; j<n; j++)
        {
            if(mat[i][j]!=0)
            {
                t_aretes[k][0]=i+1;
                t_aretes[k][1]=j+1;
                t_aretes[k][2]=mat[i][j];
                k++;
            }
        }

    }

    //printf("%d\n", nbar);
    int nbcol=3;
    /*for(k=0;k<nbar;k++){
    	printf("\n");
    	for(l=0;l<nbcol;l++)
    		printf("%d\t", t_aretes[k][l]);

    	printf("\n");
    }
    printf("\n");*/



    /*--------------------triBulles----------------------------*/
    for(i = 0; i < nbar ; i++)
    {
        //i=imin;
        for(j = nbar-1; j > i; j--)
        {
            if(t_aretes[j][2] < t_aretes[j-1][2] )
            {
                //imin = j;
                aux1=t_aretes[j-1][0] ;
                aux2=t_aretes[j-1][1] ;
                aux3=t_aretes[j-1][2] ;

                t_aretes[j-1][0]=t_aretes[j][0] ;
                t_aretes[j-1][1]=t_aretes[j][1] ;
                t_aretes[j-1][2]=t_aretes[j][2] ;


                t_aretes[j][0]=aux1;
                t_aretes[j][1]=aux2;
                t_aretes[j][2]=aux3;

            }
        }
    }

    for(i=0; i<nbar; i++)
    {
        printf("\n");
        for(j=0; j<nbcol; j++)
            printf("%d\t", t_aretes[i][j]);

        printf("\n");
    }
    printf("\n");



    /*-------------------------Kruskal----------------------------*/

    int poid=0;
    int sommet;

    etiq=(int*)malloc(n*sizeof(int)); // on alloue de la mémoire pour le tableau d'étiquettes
    for(i=0; i<n; i++)
        etiq[i]=i;

    //int k = 0;
    while(k <nbar-1)
    {
        for(i = 0; i < nbar; i++)
        {
            for(j  = i + 1; j < nbar; j++)
            {
                if(etiq[t_aretes[i][0]] != etiq[t_aretes[j][1]])
                {
                    //etiq[t_aretes[i][1]] = etiq[t_aretes[j][1]];
                    etiq[t_aretes[j][1]]= etiq[t_aretes[i][0]] ;
                    poid += etiq[t_aretes[j][1]];
                }
                else
                    continue;

            }
        }
    }

    for(i=0; i<nbar; i++) //affichage du tableau d'étiquettes
    {
        printf("\n");
        for(j=0; j<nbcol; j++){
            printf("%d\t", etiq[t_aretes[j][1]]);
        }
        printf("\n");
    }

    printf("\n");
    printf("Le coup est de : %d\n", poid); //affichage du poid || coup... ou en anglais la somme de "MST = Minimum Spanning Tree"




    /*------------------------ dijkstra---------------------------*/
	
	//!\ mise à jour dans la prochaine version




    return 0;

}
