
// C program to perform histogram equalisation to adjust contrast levels 
  
// All the needed library functions for this program 
#include <fcntl.h> 
#include <math.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
  
void histogramEqualisation(int cols, int rows,char* input_file_name, char* output_file_name);
   
// driver code 
int main() 
{ 
    // declaring variables 
    char* input_file_name; 
    char* output_file_name; 
    int cols, rows; 
  
    // defining number of rows and columns in an image 
    // here, image size is 512*512 
    cols = 512; 
    rows = 512; 
  
    // defining input file name (input image name) 
    // this boat_512_512 is a raw grayscale image 
    input_file_name = "boat_512_512.png";  //NOT WORKING ??
  
    // defining output file name (output image name) 
    output_file_name = "boat_512_512_histogram_equalised.png"; //NOT WORKING
  
    // calling function to do histogram equalisation 
    histogramEqualisation(cols, rows, input_file_name, output_file_name); 
  
    return 0; 
} 

// Function to perform histogram equalisation on an image 
// Function takes total rows, columns, input file name and output 
// file name as parameters 
void histogramEqualisation(int cols, int rows, 
                           char* input_file_name, char* output_file_name) 
{ 
    // creating image pointer 
    unsigned char* image; 
  
    // Declaring 2 arrays for storing histogram values (frequencies) and 
    // new gray level values (newly mapped pixel values as per algorithm) 
    int hist[256] = { 0 }; 
    int new_gray_level[256] = { 0 }; 
  
    // Declaring other important variables 
    int input_file, output_file, col, row, total, curr, i; 
  
    // allocating image array the size equivalent to number of columns 
    // of the image to read one row of an image at a time 
    image = (unsigned char*)calloc(cols, sizeof(unsigned char)); 
  
    // opening input file in Read Only Mode 
    input_file = open(input_file_name, O_RDONLY); 
    if (input_file < 0) { 
        printf("Error opening input file\n"); 
        exit(1); 
    } 
  
    // creating output file that has write and read access 
    output_file = creat(output_file_name, 0666); 
    if (output_file < 0) { 
        printf("Error creating output file\n"); 
        exit(1); 
    } 
  
    // Calculating frequency of occurrence for all pixel values 
    for (row = 0; row < rows; row++) { 
        // reading a row of image 
        read(input_file, &image[0], cols * sizeof(unsigned char)); 
  
        // logic for calculating histogram 
        for (col = 0; col < cols; col++) 
            hist[(int)image[col]]++; 
    } 
  
    // calulating total number of pixels 
    total = cols * rows; 
  
    curr = 0; 
  
    // calculating cumulative frequency and new gray levels 
    for (i = 0; i < 256; i++) { 
        // cumulative frequency 
        curr += hist[i]; 
  
        // calculating new gray level after multiplying by 
        // maximum gray count which is 255 and dividing by 
        // total number of pixels 
        new_gray_level[i] = round((((float)curr) * 255) / total); 
    } 
  
    // closing file 
    close(input_file); 
  
    // reopening file in Read Only Mode 
    input_file = open(input_file_name, O_RDONLY); 
  
    // performing histogram equalisation by mapping new gray levels 
    for (row = 0; row < rows; row++) { 
        // reading a row of image 
        read(input_file, &image[0], cols * sizeof(unsigned char)); 
  
        // mapping to new gray level values 
        for (col = 0; col < cols; col++) 
            image[col] = (unsigned char)new_gray_level[image[col]]; 
  
        // reading new gray level mapped row of image 
        write(output_file, &image[0], cols * sizeof(unsigned char)); 
    } 
  
    // freeing dynamically allocated memory 
    free(image); 
  
    // closing input and output files 
    close(input_file); 
    close(output_file); 
} 
