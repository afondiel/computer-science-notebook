#include "functions.h"

/***************************fUNCTIONS BUDIES***************************/

void TAB_init(float tab[], int dim){
	int i;
	for(i = 0; i < dim; i++){
		tab[i] = 0.0;
	}
}

void TAB_display(float tab[], int dim){
	int i;
	for(i = 0; i < dim; i++){
		printf("%f  ", tab[i]);	
	}
	printf("\n");	
}

void TAB_fill_increase(float tab[], int dim){
	int i;
	for(i = 0; i < dim; i++){
		tab[i]=i+1;
	}
}
void TAB_fill_random(float tab[], int size){
	int i;
	float n;
	for(i = 0; i < size; i++){
		n = rand()%(100-1+1)+1;
		tab[i] = n;
	}
}

void TAB_copy(float tab_dest[], float tab_source[], int dim){
	int i;
	for(i = 0; i < dim; i++){
		
			tab_dest[i]=tab_source[i];
	}

}  

float TAB_sum(float tab[], int dim){
	int i;
	float sum=0.0;
	for(i = 0; i < dim; i++){
		sum=sum+tab[i];
	}
	return sum;
}

float TAB_mean(float tab[], int dim){
	float mean=0.0;
	mean=TAB_sum(tab, dim)/dim;
	
	return mean;	

}
  
  float TAB_min(float tab[], int dim){
	int i;
	float min;
	min = tab[0];
	for(i = 0; i < dim; i++){
		if(min>tab[i]){
			min=tab[i];
		}
	}
	return min;
  }
  
 int TAB_index_min(float tab[], int dim){
	int i;
	for(i = 0; i < dim; i++){
		if(tab[i]==TAB_min(tab,dim))
		break;
		}
	return i;
}

int TAB_index_min_range(float tab[], int dim, int start_index, int stop_index){
		
	int i,index;
	float min;
	min = tab[start_index];
	index = start_index;
	
	for(i = start_index; i < stop_index; i++){ 
		if(min>tab[i]){
			min=tab[i];
		}
	}
	for(i = start_index; i < stop_index; i++){ 
		if(tab[i]==min){
			index=i;
		}
	}
	return index;	
	
}
			
void TAB_switch(float tab[], int dim, int index1, int index2){
	int tmp;
	tmp=tab[index1];
	tab[index1]=tab[index2];
	tab[index2]=tmp;
}
void TAB_sort_asc(float tab[], int dim){
	int tmp;
	int i,j;
	for(i=0;i<dim-1;i++){
		for(j=i+1;j<dim;j++){
			if(tab[i]>tab[j])
				tmp=tab[i];
				tab[i]=tab[j];
				tab[j]=tmp;
		}
	}
}

void TAB_sort_desc(float tab[], int size){
	


}
		 

