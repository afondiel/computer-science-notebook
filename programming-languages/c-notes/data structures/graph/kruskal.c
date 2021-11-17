#include <stdio.h>
#include <stdlib.h>



int main(){

    int choix, i,j,k,a,b,c,u,v,w,n,ne = 1;
    FILE *f;
    int min, poidmin = 0, poid[7][7], sommet[7];
    int find(int);
    int uni(int,int);

    f=fopen("kruskal.txt","r");
    fscanf("%d",&n);
    for(i = 1;i < n ; i++){
        for(j = 1;j<=n;j++){
            fscanf("%d",&poid[i][j]);
            if(poid[i][j]==0)
                poid[i][j]==100;
        }
    }

    for(i = 1,min = 100;i < n ; i++){
        for(j = 1;j<=n;j++){
            if(poid[i][j]<min)
               min= poid[i][j]
               poid[i][j] = ??
               poid[i][j]=min;
        }
    }







    return 0;

}



















