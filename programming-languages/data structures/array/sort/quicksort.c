 void exchange(int *a, int *b) 
 { 
	 int temp; 
	 temp = *a; 
	 *a = *b; 
	 *b = temp;
 } 
 
  /* Implementation of Quick Sort 
 arr[] --> Array to be sorted 
 si --> Starting index 
 ei --> Ending index */
int partition(int arr[], int si, int ei) 
{ 
	 int x = arr[ei]; 
	 int i = (si - 1); 
	 int j; 
	 for (j = si; j <= ei - 1; j++) 
	 { 
		 if(arr[j] <= x) 
		 { 
			 i++; 
			 exchange(&arr[i], &arr[j]); 
		 } 
	 } 
	exchange (&arr[i + 1], &arr[ei]); 
	return (i + 1); 
} 
 
 
 void quickSort(int arr[], int si, int ei) 
 { 
	int pi; /* Partitioning index */ 
	if(si < ei) 
	{ 
		pi = partition(arr, si, ei); 
		quickSort(arr, si, pi - 1); 
		quickSort(arr, pi + 1, ei);
	}
	
 } 