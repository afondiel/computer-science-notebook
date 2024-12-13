#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main (void){
/**************************** Declaration des variables ***********************************************************************************/

	FILE *image = NULL;
	FILE *histo = NULL;
	FILE *binaire = NULL;
	FILE *contrast = NULL;

	char NomFichier[50];

	int i, j, k, l, tmp, valeur, nbc, nbl, num;

	int N1 = 0;
	int N2 = 0;
	int S = 0;
	
	int SOMME = 0;
	int VALEUR = 0;
	int Maximum = 0;
	int Minimum = 0;
	int index_int = 0;
	int index_ext = 0;

	int h[256] = {0,};
	int hist[256] = {0,};

	float M = 0;
	float M1 = 0;
	float M2 = 0;
	float M3 = 0;
	float M4 = 0;
	float VALEUR_INTERIEUR = 0;
	float VALEUR_EXTERIEUR = 0;
	
	double F = 0;
	double Max = 0;

/**************************** Choix de l'Image ********************************************************************************************/

	printf("Choisir un image :\n1:fleur.lena\n2:navette.lena\n3:loup.lena\n4:moulin.lena\n5:table.lena\n6:muscMC\n\n");
	scanf("%d", &num);
	
	switch (num){
	
		case 1:
			strcpy(NomFichier,"fleur.lena");
		break;
	
		case 2:
			strcpy(NomFichier,"navette.lena");
		break;

		case 3:
			strcpy(NomFichier,"loup.lena");
		break;

		case 4:
			strcpy(NomFichier,"moulin.lena");
		break;

		case 5:
			strcpy(NomFichier,"table.lena");
		break;
		
		case 6:
			strcpy(NomFichier,"muscMC.lena");
		break;
	}

/**************************** Ouverture de l'Image ****************************************************************************************/
	
	image = fopen(NomFichier, "r");

   	if(image == NULL){
		printf("impossible d'ouvrir l'image");
		return 0;
	}
	
	fscanf(image, "%d %d", &nbc, &nbl);	//Recuperation de l'entête
    	fseek(image, 256, SEEK_SET); 	//Déplacement de la tête de lecture de 0 à 256

/************************* Recuperation des donnees ***************************************************************************************/

	for(i = 0 ; i < nbc*nbl ; i++){
       		valeur = (unsigned char)fgetc(image);
       		h[valeur]++;
     	}  

/************************* Histogramme  ***************************************************************************************************/

	histo = fopen("histo.lena", "wb");
	
        for(i = 0 ; i < 256 ; i++){
        	tmp = (unsigned char)fgetc(image);	//On copie les donnees de l'image
        	fputc(tmp, histo);
        }
      
	tmp = h[0];
	for(i = 1 ; i < 256 ; i++) 	if(h[i] > tmp) 	tmp = h[i]; 	//Recherche du Maximum
	
	for(i = 0 ; i < 256 ; i++)	hist[i] = (h[i]*nbc) / tmp;	//Produit en croix

        for(i = nbl ; i > 0 ; i--)	for(j = 0 ; j < nbc ; j++){
		    if(i > hist[j])	fputc(255, histo);	//On la transforme en noir et blanc
		    else	fputc(0, histo);
		}     
     
/************************* Calcul de S  ***************************************************************************************************/

	for(i = 0 ; i < 256 ; i++) M += (i*h[i]);
	
	for(j = 0 ; j < 255 ; j++){
		
		N2 = N1 = 0;
		M2 = M1 = 0;

		for(i = 0 ; i <= j ; i++){
			N1 += h[i]; 
			M1 += (i*h[i]);
		}

		N2 = (nbc*nbl) - N1;
		
		M2 = M - M1;
		
		if((N1 != 0) && (N2 != 0)){	
			M1 = M1 / N1;
	
			M2 = M2 / N2;
		
			F = (N1 * N2) * ((M1 - M2) * (M1 - M2));

			if(F > Max){
				Max = F;
				S = j;
				M3 = M1;
				M4 = M2;
			}
		}
		
	}

/************************* Affichage de l'image binaire************************************************************************************/

	binaire = fopen("image.lena", "wb");
	fseek(image , 0 , SEEK_SET);
 
        for(i = 0 ; i < 256 ; i++){
        	tmp = (unsigned char)fgetc(image);	//On copie les donnees de l'image
        	fputc(tmp, binaire);
        }

        fseek(image, 256, SEEK_SET); 	//Déplacement de la tête de lecture de 0 à 256
	fseek(binaire, 256, SEEK_SET); 	//Déplacement de la tête de lecture de 0 à 256

	for(i = 0 ; i < (nbc*nbl) ; i++){
		tmp = (unsigned char)fgetc(image);
		if(tmp > S)	fputc(255, binaire);	//En la transforme en noir et blanc
	    	else	fputc(0, binaire);
	}
	
/************************* Tableau pour le contraste **************************************************************************************/

	fseek(image, 0, SEEK_SET); 	//Déplacement de la tête de lecture de 256 à 0
	fseek(binaire, 0, SEEK_SET); 	//Déplacement de la tête de lecture de 256 à 0
	
	int tab_binaire[nbc][nbl];
	int tab_image[nbc][nbl];

	for(i = 0 ; i < nbl ; i++){
		for(j = i ; j < nbc ; j++){
			tab_binaire[j][i] = fgetc(binaire);
			tab_image[j][i]  = fgetc(image);
		}
	}

	index_ext = 0;
	index_int = 0;

	for(i = 1 ; i < (nbl-1) ; i++){
		for(j = i ; j < (nbc-1) ; j++){
			SOMME = tab_binaire[j-1][i-1] + tab_binaire[j-1][i] + tab_binaire[j-1][i+1] + tab_binaire[j][i-1]+ tab_binaire[j][i]+ tab_binaire[j][i+1] + tab_binaire[j+1][i-1] + tab_binaire[j+1][i] + tab_binaire[j+1][i+1];
			SOMME /= 9;
			Maximum = 0;
			Minimum = 255;
			
			for(k = (i-1) ; k < (i+2) ; k++){
				for(l = (j-1) ; l < (j+2) ; l++){
					if(tab_image[l][k] < Minimum) Minimum = tab_image[l][k];
					if(tab_image[l][k] > Maximum) Maximum = tab_image[l][k];
				}
			}

			VALEUR = Maximum - Minimum;

			if((SOMME == 0) || (SOMME == 255)){
				VALEUR_INTERIEUR += VALEUR;
				index_int++;
			}
			else{
				VALEUR_EXTERIEUR += VALEUR;
				index_ext++;
			}

		}	
	}

	VALEUR_INTERIEUR /= index_int;
	VALEUR_EXTERIEUR /= index_ext;
	
	printf("INDEX INTERIEUR : %d\nINDEX EXTERIEUR : %d\nCONTRASTE INTERIEUR : %.2f\nCONTRASTE EXTERIEUR : %.2f\n", index_int, index_ext, VALEUR_INTERIEUR, VALEUR_EXTERIEUR);
   
/************************* Fermeture du programme *****************************************************************************************/

	close(image);
	close(binaire);
	close(histo);
	close(contrast);


/************************* Affichage l'histogramme ****************************************************************************************/

	system("xv image.lena");

	return 0;
}
