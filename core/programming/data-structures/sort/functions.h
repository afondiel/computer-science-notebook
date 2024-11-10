#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define SIZE 10

/************ prototypes des fonctions**********/

void TAB_init(float tab[], int dim);

void TAB_display(float tab[], int dim);

void TAB_fill_increase(float tab[], int dim);	
  
void TAB_fill_random(float tab[], int size);

void TAB_copy(float tab_dest[], float tab_source[], int dim);	
  
float TAB_sum(float tab[], int dim);	
  
float TAB_mean(float tab[], int dim);	
  
float TAB_min(float tab[], int dim);
	
int TAB_index_min(float tab[], int dim);	

int TAB_index_min_range(float tab[], int dim, int start_index, int stop_index);

void TAB_switch(float tab[], int dim, int index1, int index2);

void TAB_sort_asc(float tab[], int dim);	

void TAB_sort_desc(float tab[], int size);	
  

  

 
	
  




















#endif
