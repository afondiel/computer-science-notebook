#include <stdio.h>
#include <stdlib.h>


typedef struct arbre arbre;
struct arbre{
	int v;
	arbre *g;
	arbre *d;
};

arbre *inserer (arbre *a, int c);
void afficher (arbre *a);



int main(){
	
	arbre *a;
	int c=1;
	a=NULL;	
	printf("------------------------------------------------------\n");
	while(c>0){
		//printf("\n");
		printf("Entrer une valeur (taper 0 pour sortir)?");
		scanf("%d",&c);
		if(c>0)
			a=inserer(a,c);
	}
	printf("\n");
	//couper(a, &G, &D, c);
	afficher(a);
	printf("---------------------------------------------------------\n");
	
	return 0;


}


arbre *inserer (arbre *a, int c){
	if (a==NULL){
		a=(arbre*)malloc(sizeof(arbre));
		a->v=c;
		a->g=NULL;
		a->d=NULL;
	}
	else{
		if(c>a->v)
			a->d=inserer(a->d,c);
		else 
			a->g = inserer(a->g,c);
	}
	return a;
}
	
void afficher (arbre *a){
	if(a==NULL)
	return; 
	afficher(a->g);
	printf("%d(",a->v);
	if(a->g!=NULL)
		printf("%d-",a->g->v);
	else printf("-");
	if(a->d!=NULL)
		printf("%d)",a->d->v);
	else printf(")");
	printf("\n");
	afficher(a->d);
	//printf("\n");

}			

void couper(arbre *a, int c, arbre **G, arbre **D, ){
	
	if (a==NULL){
		*G=NULL;
		*D=NULL;
		return; 
	}
    else{
		if(c < a->v){
			*D = a;
			couper(a->g, c, G, &((*D)->g)); //

    	}
		else{
			*G = a;
			couper(a->d, c, &((*G)->d), D); //
		}
			
		
			
	}
}
   













	
	
	
	
	
	
	
	

	
	
	
	
	
	
	
	
	




