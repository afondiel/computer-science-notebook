/********************************************************
* hello -- program to print out "Hello World".          *
* Not an especially earth-shattering program.           *
*                                                       *
* Author: Afondiel                                      *
*                                                       *
* Purpose: Demonstration of a simple program.           *
*                                                       *
* Usage:                                                *
* Runs the program and the message appears.             *
********************************************************/
#include <stdio.h>
#include <stdlib.h>

//========= C Language interview question =========
// ## Big Picture ##
//- Programming paradigm
//- Design patterns
//
//- Language :
//=> Data types
//=> Variables
//=> Conditions
//=> Functions
//=> Advanced types : struct, union, pointer, volatile, static...
//=> Modular programming
//=> Object based programming
//
//- Data structure
//=> Big O
//=> Table
//=> Sort
//=> Stack
//=> Queue
//=> Tree
//=> Graph


int main()
{

/*****************************************************************
 *  From Pratical C Programming - 3rd edition - Steve Oualline
******************************************************************/
// Chap I - Basics
// I.1-What's C
// I.2-Basics of Program Writing
// I.3-Style
// I.4-Basic Declarations and Expressions
// I.5-Arrays, Qualifiers, and Reading Numbers
// I.6-Decision and Control Statements
// I.7-Programming Process

// Chap II - Simple Programming
// II.8-More Control Statements
// II.9-Variable Scope and Functions
// II.10-C Preprocessor
// II.11-Bit Operations
// II.12-Advanced Types
// II.13-Simple Pointers
// II.14-File Input/Output
// II.15-Debugging and Optimization
// II.16-Floating Point

// Chap III - Advanced Pointers
// III.17-Advanced Pointers
// III.18-Modular Programming
// III.19-Ancient Compilers
// III.20-Portability Problems
// III.21-C's Dustier Corners
// III.22-Putting It All Together
// III.23-Programming Adages
// III.17-Advanced Pointers
// Chap IV - Other Languages Features
// Chap IV-A. ASCII Tables
// Chap IV-B. Ranges and Parameter Passing Conversions
// Chap IV-C. Operator Precedence Rules
// Chap IV-D. A Program to Compute a Sine Using a Power Series

//------------------------------- Coding -------------------------------//
//=========================//
//      I.1-What's C       //
//=========================//
    printf("I.1-What\'s C ?\n");

// - c allows a sw enginer to communicate w/ a computer
// - c is a highly flexible and adaptable language
// - Applications : firmware for mcu, os, apps, graphics programming
// - c vs C++ battle :
//   => c++ was added class and facilitate code reuse
//   => c++ used for automation
//   => c better for embedded control applications (back in 1997)
// HOW PROGRAMMING WORKS ?
// - machines “think” in numbers, people don’t
// - assembly => Translation => Machine Language

// HISTORY :
// - Before C were : FORTRAN, COBOL, PASCAL, B....
// - created in 1970 by Denis Richie => Goal Writing OS
// - 1980 Bjarne Stroustrup created C++
// - the JAVA was born from C++ to handle bugs.

// HOW C Works ?
// - Compiler => REVOLUTION
// - C LANGUAGE => (COMPILER | assembler |LINKER ) => Machine Language
// - Programs divided : Data + instrucions

// HOW TO LEARN C ?
// - only way : write program + Debbuging
// - Keep good programming style
// - comments helps to organize thoughts even before writing CODE/PROGRAM
// - Think portability
//

//======================================//
//  I.2-Basics of Program Writing       //
//======================================//
    printf("I.2-Basics of Program Writing\n");

//<-- Compilation Process/tools -->
// 1. Pre-processing
// 2. Compilation
// 3. Assembling
// 4. Linkage
// compilation : gcc file.c -g –o file.o (linux)
// compilation : gcc file.c -g –o file.exe (windows)
// - g : enables debugging
// - Wall : enables warning
//

//====================//
//  I.3 - Style      //
//===================//
    printf("I.3-Style\n");

//
//Common Coding
//Practices
//Coding Religion//Paradigm
// - structured programming
// - top-down programming
// - goto-less programming
// - object oriented design (OOD)

//Indentation and
//Code Format
//Clarity
//Simplicity
//Summary :
// - program should be concise and easy to read.
// - reference work describing the algorithms and data used inside it
// - Everything should be documented with comments


//================================================//
//  I.4-Basic Declarations and Expressions      //
//===============================================//
    printf("I.4-Basic Declarations and Expressions \n");
//Agenda
//Elements of a Program
//Basic Program Structure
// => the data declarations, functions, and comments
//Simple Expressions
    printf("Simple Expressions : %d\n", (1 + 2) * 4);    /* Operations*/

//Variables and Storage
//Variable Declarations
//Integers
//Assignment Statements
//printf Function
//Floating Point
//Floating Point Versus nteger Divide
//Characters
//Answers
//Programming
//Exercises



























    return 0;
}
