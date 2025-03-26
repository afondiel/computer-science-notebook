#include <stdio.h>
#include <stdlib.h>



int main(){


	FILE *p;
	int **mat,**t_arretes;
	int i,j,k,l,m,aux;
	int n=6;
	
	p=fopen("graphe.txt","r");
	fscanf(p,"%d", &n);
	
	mat=(int**)malloc(n*sizeof(int*));
	for(i=0;i<n;i++)
		mat[i]=(int*)malloc(n*sizeof(int));
	
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			fscanf(p,"%d", &mat[i][j]);	
	
		}
	} 
	/*for(k=0;k<n;k++){ // affichage de la matrice d'adjacence 
		printf("\n");
		for(l=0;l<n;l++)
			printf("%d\t", mat[k][l]);
			printf("\n");
	}*/
				
	/*for(k=0;k<n;k++){				
		printf("\n");
		for(l=k+1;l<n;l++)
			printf("%d\t", mat[k][l]);
			printf("\n");	
	}*/
	/*--------------- Allocation dynamiques ------------------*/
	t_arretes=(int**)malloc(n*sizeof(int*)); 
	for(i=0;i<n;i++)
		t_arretes[i]=(int*)malloc(n*sizeof(int));
	
	int nbar=0;
	for( k = 0; k < n; k++){				
		for( l = k; l < n; l++){
			if(t_arretes[k][l] != 0){
				t_arretes[nbar][0] = k+1;
		   		t_arretes[nbar][1] = l+1;
				t_arretes[nbar][2] = mat[k][l];
				nbar++;

				//printf("%d %d %d\n", k+1,l+1,mat[k][l])
			}
			
		}
	}
	
	/*-------------------tri par insertion du tableau d'arrete "t_arretes" --------------------*/
	for(k=1;k<n;k++){				
		for( l = k; l > 0; l++){
			if(t[l] < mat[l-1] ){
				aux = t[l] ;
				t[l] = t[l-1] ;
				t[l-1] = aux;
									
			}
	
	
		}
	
	}
	
	

		
	
	return 0;
}
