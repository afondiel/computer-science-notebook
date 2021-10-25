//#include <iostream>
//#include <vector>
//#include <string>

//#define SIZE 10

//using namespace std;

//int main(int argc, char *argv[]) {return (0);}

		/********************************************************
		 ********************************************************
		 ********										  *******
		 ********		   Part I. The Basics		      *******
		 ******** 										  *******
		 ********************************************************
		 ********************************************************/
	

	/*--------------------------------------------------------------/
	/							 CHAP 1 : What Is C++?				/
	/--------------------------------------------------------------*/
	/* - A brief history of C++
	 * - C++ organization 
	 * - How to learn C++ ? */
	/*
	//DATA : 
	int total;				// Total number accounts
	int balance[100];		// Balance (in cents) for all 100 accounts

	struct rectangle {
		int width;			// Width of rectangle in pixels
		int height;			// Height of rectangle in pixels
		color_type color;	// Color of the rectangle
		fill_type fill;		// Fill pattern
	};
	//INSTRUCTIONS : 
	area = (base * height) / 2.0; // Compute area of triangle

	*/

	 /*-----------------------------------------------------------------/
	 /				 CHAP 2: The Basics of Program Writing				/
	 /-----------------------------------------------------------------*/
	/*
		++ IDE
		++ "Wrappers" programs?? (Programs Layers? Midle programs?? EG : MAKE cmd? )
		++ From conception to execution
			>>High-level code>>Compiler>>Assembly Language Program>>Assembler>>Object code>>Linker+Library>>Executable Program
		GNU compiler : g++ -g -Wall -o program.exe sourceFile.cpp 
		++Other compiler : Barland, Turbo, generic UNIX ...
	*/
		
	 /*-----------------------------------------------------------------/
	 /				 CHAP 3: The STYLE								   /
	 /-----------------------------------------------------------------*/

		/* Long comment section */

		// short comment

		//Variable Declaration : 
	//Bad style 
	//int p, q, r;

	//BadASS style ;)
	//int account_number;		// Index for account table
	//int balance_owed;		// Total owed us (in pennies)		
		
	//Avoid complex syntaxes
	//A function should not be longer than 3 pages
	//A file longer than 1500 lines of codes >> Modular programming ? 
	//1 class per module
	//Pick one style when work in a group to maintain Consistency


	/*-----------------------------------------------------------------/
	/			CHAP 4 : Basic Declarations and Expressions		      /
	/-----------------------------------------------------------------*/

	//Program = DATA(Variables) + INSTRUCTIONS (Code) 
	// (SAM != sam)  : True
	//!\ reserve words(int, float, while, for ...) : can not be used as Variables 
	// Declaration : type name; // comment
	//int answer;
	//cout<<"Hello world"<<endl;
	//cout << "Half of " << 64 << " is " << (64 / 2) << "\n";
	//cout << "The value of 1/3 is " << answer << "\n";
	//Operations : +,-,*,/,% ...
	//Types : char(single character), int, float, short, long ...
	//escape character : \  ? 

	/*cout << "*** "<< "\t" << "***" << "\t" << "**********" << "\t" << "***" << "\t\t"		  << "***" << "\t\t" << "***********" << endl;
	cout << "***" << "\t" << "***" << "\t" << "**********" << "\t" << "***" << "\t\t"		  << "***" << "\t\t" << "***********" << endl;
	cout << "***" << "\t" << "***" << "\t" << "***"		   << "\t\t" << "***" << "\t\t"		  << "***" << "\t\t" << "***" << "\t" << "***" << endl;
	cout << "***" << "\t" << "***" << "\t" << "**********" << "\t" << "***" << "\t\t"		  << "***" << "\t\t" << "***" << "\t" << "***" << endl;
	cout << "***********"		   << "\t" << "**********" << "\t" << "***" << "\t\t"		  << "***" << "\t\t" << "***" << "\t" << "***" << endl;
	cout << "***********"		   << "\t" << "**********" << "\t" << "***" << "\t\t"		  << "***" << "\t\t" << "***" << "\t" << "***" <<endl;
	cout << "***" << "\t" << "***" << "\t" << "***"		   << "\t\t" << "***" << "\t\t"		  << "***" << "\t\t" << "***" << "\t" << "***" << endl;
	cout << "***" << "\t" << "***" << "\t" << "**********" << "\t" << "**********" <<"\t" << "**********" << "\t" << "***********" << endl;
	cout << "***" << "\t" << "***" << "\t" << "**********" << "\t" << "**********" <<"\t" << "**********" <<"\t" << "***********" << endl;*/

	/*-----------------------------------------------------------------/
	/			CHAP 5 : Arrays, Qualifiers, and Reading Numbers	   /
	/-----------------------------------------------------------------*/
	

	//// Arrays 
	///int array[SIZE]				// SIZE : is a define
	// List of data to be sorted and averaged 
	// int data_list[3]; ???
	//	>>ARRAYS fonction : 
	//sizeof(array)
	//malloc ?  : for dynamic array

	//// Strings : Arrays of chars

	//char name[] = {'S', 'a', 'm', '\0'};
	//char string[size] = "hello";
	//		>>STR Fonctions : 
	//strcpy(dest,src) : to cpy a string into another
	//strcat(string1, string2) Concatenates string2 onto the end of string1
	//length = strlen(string) Gets the length of a string
	//strcmp(stringl, string2) 0 if string1 equals string2; otherwise, nonzero
	
	//READING DATA  : 
	//char pixel;
	// cin>>pixel;							//read a char, number
	// cin>>pixel;				 cin.ignore();			//To ignore asking the user or to pause the program ? 	
	
	
	//cin.getline(string, sizeof(string));			//read a str
	//char name[100]; // The name of a person
	//cin.getline(name, sizeof(name));
	
	//MULTIDIMMENSION ARRAYS : 

	//int matrix[row_X][colomn_Y];
	//four_dimensions[10][12][9][5];

	//int matrix[2][4] =
	//{
	//{1, 2, 3, 4},
	//{10, 20, 30, 40}
	//};

	//This is shorthand for:
	//matrix[0][0] = 1;
	//matrix[0][1] = 2;
	//matrix[0][2] = 3;
	//matrix[0][3] = 4;
	//matrix[1][0] = 10;
	//matrix[1][1] = 20;
	//matrix[1][2] = 30;
	//matrix[1][3] = 40;
	
	//\\//\\ TYPES //\\//\\
	// char, int, long, float, double
	// signed long int answer;	      //long int answer					// SAME result
	// SIGNED char ch;				  // Very short integer // Range is -128 to 127 (Now char became an a short INTEGER!)
	// Reference :  int cnt; => int &cntRef = ref; ? 
	// Qualifiers : Adjectif(of) + TYPES??
	// volatile : I/O drivers + Shares memory applications + values that change anytime ? 
	// variables++ VS ++variables ? if variables == 5 => 1 : 5; 2 : 6 

	/*int var, res1, res2;
	var = 5;
	res1 = var++;

	res2 = ++var;

	cout << "Res1 : " << res1 << "\tRes2 : " << res2 << endl;*/

	/*-----------------------------------------------------------------/
    /			CHAP 6 : Decision and Control Statements	           /
    /-----------------------------------------------------------------*/

	// Decisions ? 
	// if, else, elsif ? goto :(  ?  
	//if (condition)
		//statement;
	// Looping ? 
	//while (condition)
		//statement;
	//fn = fn-1 + fn-2 (Suite de Fibonacci ?)
	//next_number = current_number + old_number;
	/*
	//Suite de Fibonacci : less than 100 ?
	int old_number; // previous Fibonacci number
	int current_number; // current Fibonacci number
	int next_number; // next number in the series

	// start things out
	old_number = 1;
	current_number = 1;
	cout << "0\n"; // Print first number
	while (current_number < 100) {
		cout << current_number << '\n';
		next_number = current_number + old_number;
		old_number = current_number;
		current_number = next_number;
	}
	*/
	//Break : Exit the loop 
	//Continue : to jump a statement ?

	/*-----------------------------------------------------------------/
	/			CHAP 7 : Programming Process				           /
	/-----------------------------------------------------------------*/
	//It's just a simple matter of programming.
		//—Any Boss Who Has Never Written a Program
	
		//Makefile : make uses the modification dates of the files to determine whether or not a compilation is necessary.
	/*[File:calcl / makefile.msc]
	#
	# Makefile for Microsoft Visual C++
	#
	CC = cl
	#
	# Flags
	# AL -- Compile for large model
	Page 105
	# Zi -- Enable debugging
	# W1 -- Turn on warnings
	#
	CFLAGS = / AL / Zi / W1
	all : calc.exe
		calc.exe : calc.cpp
		$(CC) $(CFLAGS) calc.cpp
		clean :
	erase calc.exe*/
	
	//@TODO : Programming exercises ?

		/********************************************************
		 ********************************************************
		 ********										  *******
		 ********		Part II. Simple programming	      *******
		 ******** 										  *******
		 ********************************************************
		 ********************************************************/


	/*-----------------------------------------------------------------/
	/			CHAP 8 : More Control Statements			           /
	/-----------------------------------------------------------------*/
		
	// for (initial-statement; condition; iteration-statement) 
		//body - statement; 

	/*initial-statement;
	while (condition) {
	body-statement;
	iteration-statement;
	}*/

	/*switch (expression)
		case constantl:
		statement
		....
		break;
		case constant2:
		statement
		.....
		// Fall through
		default:
		Page 121
		statement
		....
		break;
		case constant3:
		statement
		....
		break;
		}*/
	
		//@TODO :  Programming exercises?



	/*-----------------------------------------------------------------/
	/			CHAP 9 : Variable Scope and Functions			       /
	/-----------------------------------------------------------------*/
	//\\VARIABLES//\\
	//SCOPE//\\ : global, local, sous-local
	//Local variables are temporary unless they are declared STATIC(this scope means the variablesis local to in the current file)	
	//STORAGE CLASS :
	//Global var : Permanent(From the beginning To end of progrm the value remains the SAME )
	//Local var : Temporary (change throughout the program//also called as automatic variables : they allocate space in the memory automatically)
	//QUALIFIER : auto, using ? 

	//\\Functions//\\

	//Types : overload, recursive functions, default, inline(tells the compiler that the code is too short) ? 
	//Style: 
	/*******************************************
	* Triangle -- compute area of a triangle *
	* *
	* Parameters *
	* width -- width of the triangle *
	* height -- height of the triangle *
	* *
	* Returns *
	* area of the triangle *
	*******************************************/
	//Prototype : float triangle(float width, float height);
	/*float triangle(float width, float height)
	{
		float area; // Area of the triangle
		area = width * height / 2.0;
		return (area);
	}*/
	
	//CALL/RETURN type : call by value, call by refrence
	//Void functions  : int get_value(void);
	//always put "(void)" -> to say that the function is a VOID

	//const int& biggest(int array[], int n_elements); //Return by reference
	//Array Parameters

	/*int sum_matrix(int matrixl[10][10]); // Legal
	int sum_matrix(int matrixl[10][]); // Legal
	int sum_matrix(int matrixl[][]); // Illegal*/	
	
	/*int length(char string[])
	{
		//This function(when working properly) performs the same function as the library function "strlen"
			int index; // Index into the string
			/*
			* Loop until we reach the end-of-string character
			*/
		/*for (index = 0; string[index] != '\0'; ++index)
			/* do nothing */
			/*return (index);
	}*/	
	/*
	int square(int value) {
	return (value * value);
	}
	We also want to square floating point numbers:
	float square(float value) {
	return (value * value);
	}*/
	//A GENERIC FUNCTION (template) could a solution to function overloading ??!
	
	//RECURSION : 1. It must have an ending point.| 2. It must make the problem simpler.
	
	/*A definition of factorial is:
	fact(0) = 1
	fact(n) = n * fact(n-l)
	In C++ this is:
	int fact(int number)
	{
	if (number == 0)							//Ending point?
	return (1);
	//else
	return (number * fact(number - 1));			//It simplifies the problem
	}*/

	//Another exemple : 

	/*In C++ this is:
	int sum(int first, int last, int array[])
	{
		if (first == last)
			return (array[first]);
		//else
		return (array[first] + sum(first + 1, last, array));
	}

	For example :
	Sum(1 8 3 2) =
	1 + Sum(8 3 2) =
	8 + Sum(3 2) =
	3 + Sum(2) =
	2
	3 + 2 = 5
	3 + 2 = 5
	8 + 5 = 13
	1 + 13 = 14 
	Answer : 14
	*/

	//@TODO : Programming exercises? 


	/*-----------------------------------------------------------------/
	/			CHAP 10: The C++ Preprocessor						   /	
	/-----------------------------------------------------------------*/
	//Preprocessors : are directives seen by the compiler before ENTER in the MAIN()  ??!
	//#include, #define, #ifdef?-endif ?, #pragma ? 

	// DEFINE : ALLOWS to make operations without using/charge the Computer Memory?

	//SAME :#define SIZE 20 // The array size is 20  || const int SIZE = 20; // The array size is 20
	// backslash operator: (\) to continue the line/ pass to another line
	//#define Name Substitute-Text
	//MACRO : #define FOO bar
	
	//#define FOR_ALL for (i = 0; i < ARRAY_SIZE; ++i) // bad practices
	//Good ONE : #define BIG_NUMBER 10 ** 10
	//Parameterized Macros : #define SQR(x) ((x) * (x)) /* Square a number */


	/*
	#define MAX 10 // Define a value using the pre-processor
	// (This can easily cause problems)
	const int MAX = 10; // Define a C++ constant integer
	// (Safer)
	*/
	//CONDITIONAL COMPLATION :  #ifdef-#endif
	/*
		#define DEBUG // Turn debugging on 
	
		#ifdef DEBUG
		cout << "In compute_hash, value " << value << " hash << hash <<
		"\n";
		//#else //DEBUG
		#endif /* DEBUG */
	/**/
	//INCLUDE : 
	//#include <iostream.h>/<iostream>					//in the header - files
	//#include "data.h"									// in the directory folder

	//when using a Header - files "TWICE" in the program : CHECK?
	//#ifndef _CONST_H_INCLUDED_
	/* Define constants */
	//#define _CONST_H_INCLUDED_
	//#endif /* _CONST_H_INCLUDED_ */

	/*The # Operator
	The # operator is used inside a parameterized macro to turn an argument into a string. 
	For example:
	#define STR(data) #data
	STR(hello)

	generates
	"hello"*/
	


	//@TODO :  Programming exercises?

	/*-----------------------------------------------------------------/
	/			CHAP 11: Bit Operations								   /
	/-----------------------------------------------------------------*/
	// Operations : NOT : ~, OR: |, AND: &, NOR: ~(|), NAND: ~(&), XOR: ^ , Shift left : << , Shift right : >>

	
	//i = j << 3; // Multiply j by 8 (2**3)
	//q = i >> 2; q = i / 4;

/*
	// /!\ A tester /!\
	#include <iostream.h>
	const int HIGH_SPEED = (1<<7); /* modem is running fast */
	// we are using a hardwired connection
	/*const int DIRECT_CONNECT = (1 << 8);
	char flags = 0; // start with nothing
	main()
	{
		flags |= HIGH_SPEED; // we are running fast
		flags |= DIRECT_CONNECT; // because we are wired together
		if ((flags & HIGH_SPEED) != 0)
			cout << "High speed set\n";
		if ((flags & DIRECT_CONNECT) != 0)
			cout << "Direct connect set\n";
		return (0);
	}*/

	//Bitmapped graphics : 
	
	//bit_array[0] [7] |= (0x80 >> (4) );						//Figure 11-2. Array of bits 

	//Convert bit 2 Byte cordinator:(x, y) => (x/8,y)

	/*void inline set_bit(const int x,const int y)
	{
		graphics[(x)/8][y] |= (0x80 >> ((x)%8))
	}*/
	
	//>>>>>>>>>>>>>>>>>>>> MAIN :Example 11 - 2. graph / graph.cc <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	//#include <iostream>
	//#include <vector>
	//#include <string>
	//
	//#define SIZE 10
	//
	//using namespace std;
	
	//const int X_SIZE = 40; // Size of array in the X direction
	//const int Y_SIZE = 60; // Size of array in the Y direction
	///*
	//* We use X_SIZE/8 since we pack 8 bits per byte
	//*/
	//char graphics[X_SIZE / 8][Y_SIZE]; // The graphics data
	//
	//void set_bit(const int x, const int y); // Prototype de la fonction : Convert bit to Bytes
	//
	//void print_graphics(void); // Prototype de la fonction : Print the data ?
	//int main(int argc, char *argv[]) {
	//int loc; // Current location we are setting
	////void print_graphics(void); // Print the data
	//for (loc = 0; loc < X_SIZE; ++loc)
	//	set_bit(loc, loc);
	//print_graphics();
	//return (0); 
	//}

	/* Functions Declaration */

	///******************************************************
	//* set_bit -- set a bit in the graphics array *
	//* *
	//* Parameters *
	//* x,y -- location of the bit *
	//******************************************************/
	////inline void set_bit(const int x, const int y)
	//void set_bit(const int x, const int y)
	//{
	//	graphics[(x) / 8][y] |= (0x80 >> ((x) % 8));
	//}
	//
	///*******************************************************
	//* print_graphics -- print the graphics bit array *
	//* as a set of X and .'s *
	//*******************************************************/
	//void print_graphics(void)
	//{
	//	int x; // Current x byte
	//	int y; // Current y location
	//	int bit; // Bit we are testing in the current byte
	//	for (y = 0; y < Y_SIZE; ++y) {
	//		// Loop for each byte in the array
	//		for (x = 0; x < X_SIZE / 8; ++x) {
	//			// Handle each bit
	//			for (bit = 0x80; bit > 0; bit = (bit >> 1))
	//				if ((graphics[x][y] & bit) != 0)
	//					cout << 'X';
	//				else
	//					cout << '.';
	//		}
	//	}
	//	cout << '\n';
	//}
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	//@TODO :  Programming exercises?

		/********************************************************
		 ********************************************************
		 ********										  *******
		 ******** Part III. Advanced types and classes    *******
		 ******** 										  *******
		 ********************************************************
		 ********************************************************/


	/*-----------------------------------------------------------------/
	/			CHAP 12: Advanced Types							   /
	/-----------------------------------------------------------------*/
	//Struct, enum, unions, typedef, class

	//Struct
	/*
	struct structure-name {
		field-type field-name // Comment
		field-type field-name // Comment
		....
	} variable-name;
	
	*/

	// UNIONS : A union is similar to a structure; however, it defines a SINGLE location that can be given many different field names.
	/*union value {
		long int i_value; // Long integer version of value
		float f_value; // Floating version of value
	}*/

	//TYPEDEF : 
	//typedef type - declaration
	// Ex : typedef int width; // Define a type that is the width of an object || same : #define width int
	
	//ENUM : The enumerated (enum) data type is designed for variables that can contain only a limited set
	//of values.These values are referenced by name(tag).The compiler assigns each tag an INTEGER value
	//internally, such as the days of the week.

	//typedef int day_of_the_week; // Define the type for days of the week
	//const int SUNDAY = 0;
	//const int MONDAY = 1;
	//const int TUESDAY = 2;
	//const int WEDNESDAY = 3;
	//const int THURSDAY = 4;
	//const int FRIDAY = 5;
	//const int SATURDAY = 6;
	///* Now to use it */
	//day_of_the_week today = TUESDAY;
	
	//enum day_of_the_week {SUNDAY = 0, MONDAY = 1, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY};
	/* Now use it */
	//enum day_of_the_week today = TUESDAY; 

	//enum enum-name {tag-1, tag-2, . . .} variable-name

	//!\cast or typecast operation. int (string/char) /!\/!\/!\/!\
	
	//Bit fields / Packaded structures (CHAMPS DE BITS) ////////// MEMORY OPTIMIZATION (Bit field, inline ....)
	
	//struct item {
	//	unsigned int list; // True if item is in the list
	//	unsigned int seen; // True if this item has been seen
	//	unsigned int number; // Item number
	//};
	
	/* The  field "list" and "seen" can ONLY have to values 0 / 1 => We can include the TWO bits in the field of "number field" AND gain 4 BYTES space in the memory */
	/* 2 fields USELESS => (6 - 2*2) = 2 bytes ONLY  */

	//struct item {
	//	unsigned int list : 1; // True if item is in the list
	//	unsigned int seen : 1; // True if this item has been seen
	//	unsigned int number : 14; // Item number
	//};
	
	//Arrays of Structures \\

	
	/*struct time {
		int hour; // Hour (24-hour clock)
		int minute; // 0-59
		int second; // 0-59
	};*/

	// struct time lap[MAX_LAPS];
	/*
	 * Runner just past the timing point
	 */
	//lap[count].hour = hour;
	//lap[count].minute = minute;
	//lap[count].second = second;
	//++count;
	
	//@TODO :  Programming exercises?

	
	/*-----------------------------------------------------------------/
	/			CHAP 13: Simple Classes							   /
	/-----------------------------------------------------------------*/

	//Class : object : DATA + FUNCTIONS/METHODS ( Operation/instruction on the DATA ) ? 
	

	//A STACK is an algorithm for storing data. Data can be put in the stack using a PUSH operation.
	//The POP operation removes the data. Data is stored in last - in - first - out (LIFO) orde

	// --------------------------------------------------	
	// |Type de mémoire	|	Zone	|Données	   |STM32|
	// ----------------	|-----------|----------    |-----|
	// |	ROM			|Code		| Programme	   |Code |
	// ----------------	|--------	|--------------|-----|
	// |				|Statique	| Var.globales |Data |
	// |	RAM			|-----------|--------------|-----|
	// |				|Pile		| Var. locales |STACK|
	// | ---------------|-----------|--------------|-----|
	// |				|			|			   |	 |
	// |				|TAS		|Allocation dyn|Heap |
    // ----------------	|----		|--------------|-----|

	// PUSH : ??
	// POP : ??

	//>>>>>>>>>>>>>>>>>>>>>>>> Designing a STACK : <<<<<<<<<<<<<<<<<<<<<<<<<<<<

	// const int STACK_SIZE = 100; // Maximum size of a stack

	// // The stack itself
	// struct stack {
	// 	int count; // Number of items in the stack
	// 	int data[STACK_SIZE]; // The items themselves
	// };

	//>>>>>>>> PUSH function (SETTER???) (without a stack overflow): 

	// inline void stack_push(struct stack& the_stack, const int item)
	// {
	// 	the_stack.data[the_stack.count] = item;
	// 	++the_stack.count;
	// }

	//>>>>>>>> POP function (GETTER?) : 
	// inline int stackpop(struct stack& the_stack)
	// {
	// 	// Stack goes down by one
	// 	--the_stack.count;

	// 	// Then we return the top value
	// 	return (the_stack.data[the_stack.count]);
	// }

	//inline void stack_init(struct stack &the_stack)
	//{
	//the_stack.count = 0; // Zero the stack
	//}

	//struct stack a_stack;			// Declare the stack
	//stack_init(a_stack);			// Initialize the stack
	//// Stack is ready for use !!!

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Example 13 - 1. stack_s / stack_s.cc <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//
///********************************************************
//* Stack *
//* A set of routines to implement a simple integer *
//* stack. *
//* *
//* Procedures *
//* stack_init -- initialize the stack *
//* stack_push -- put an item on the stack *
//* stack_pop -- remove an item from the stack *
//********************************************************/
//#include <iostream>
//#include <stdlib.h>
//using namespace std;
//
//const int STACK_SIZE = 100; // Maximum size of a stack
//// The stack itself
//struct stack {
//	int count;				// Number of items in the stack
//	int data[STACK_SIZE];	// The items themselves
//};
///*********************************************************
//* stack_init -- initialize the stack *
//Page 200
//Example 13-1 stack_s/stack_s cc (Continued)
//* *
//* Parameters *
//* the_stack -- stack to initialize *
//*********************************************************/
//inline void stack_init(struct stack& the_stack)
//{
//	the_stack.count = 0; // Zero the stack
//}
///*********************************************************
//* stack_push -- push an item on the stack *
//* *
//* Warning: We do not check for overflow *
//* *
//* Parameters *
//* the_stack -- stack to use for storing the item *
//* item -- item to put in the stack *
//*********************************************************/
//inline void stack_push(struct stack& the_stack, const int item)
//{
//	the_stack.data[the_stack.count] = item;
//	++the_stack.count;
//}
///*********************************************************
//* stack_pop -- get an item off the stack *
//* *
//* Warning: We do not check for stack underflow *
//* *
//* Parameters *
//* the_stack -- stack to get the item from *
//* *
//* Returns *
//* the top item from the stack *
//*********************************************************/
//inline int stackpop(struct stack& the_stack)
//{
//	// Stack goes down by one
//	--the_stack.count;
//	// Then we return the top value
//	return (the_stack.data[the_stack.count]);
//}
//// A short routine to test the stack
//int main(int argc, char *argv[])
//{
//	struct stack a_stack; // Stack we want to use
//	stack_init(a_stack);
//	// Push three values on the stack
//	stack_push(a_stack, 1);
//	stack_push(a_stack, 2);
//	stack_push(a_stack, 3);
//	// Pop the items from the stack
//	cout << "Expect a 3 ->" << stackpop(a_stack) << '\n';
//	cout << "Expect a 2 ->" << stackpop(a_stack) << '\n';
//	cout << "Expect a 1 ->" << stackpop(a_stack) << '\n';
//	return (0);
//}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Example 13 - 1 - END <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>> METHOD2 : CLASSES <<<<<<<<<<<<<<<<<<
//#include <iostream>
//#include <stdlib.h>
//using namespace std;
//
//const int STACK_SIZE = 100; // Maximum size of a stack
//
//class stack {
//	private:
//		int count; // Number of items in the stack
//		int data[STACK_SIZE]; // The items themselves
//	public:
//		// Initialize the stack
//		void init(void);
//		// Push an item on the stack
//		void push(const int item);
//		// Pop an item from the stack
//		int pop(void);
//};

// >>>>>>>>>>>>>>>>>>>>>>>> STRUCTURES VS CLASSES <<<<<<<<<<<<<<<<<<
//fields ====> class MEMBER (variables and functions).
//
//Classes HAVE Access privileges : private(cannot be seen outside of the class), public(indicates members that anyone can access), protected(it allows access by derived classes)
//Structures NOT.
//Access a fucntion in the STRUCTURES => (.) SAME with a class
//to LINK the prototype inside the class and THEIR definition outside !!! => scope operator (::) /!\ 

//inline void stack::init(void)
//{
//	count = 0; // Zero the stack
//}
//
//inline void stack::push(const int item)
//{
//	data[count] = item;
//	++count;
//}
//inline int stack::pop(void)
//{
//	// Stack goes down by one
//	--count;
//	// Then we return the top value
//	return (data[count]);
//}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Example 13 - 2. stack_c / stack_c.cc <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//
///********************************************************
//* Stack *
//* A file implementing a simple stack class *
//********************************************************/
//
//#include <stdlib.h>
//#include <iostream>
//using namespace std;
//
//const int STACK_SIZE = 100; // Maximum size of a stack
///********************************************************
//* Stack class *
//* *
//* Member functions *
//* init -- initialize the stack *
//* push -- put an item on the stack *
//* pop -- remove an item from the stack *
//********************************************************/
////The stack itself
//class stack {
//private:
//	int count; // Number of items in the stack
//	int data[STACK_SIZE]; // The items themselves
//public:
//	// Initialize the stack
//	void init(void);
//	// Push an item on the stack
//	void push(const int item);
//	// Pop an item from the stack
//	int pop(void);
//};
///********************************************************
//* stack::init -- initialize the stack *
//***** **************************************************/
//inline void stack::init(void)
//{
//	count = 0; // Zero the stack
//}
///********************************************************
//* stack::push -- push an item on the stack *
//* *
//* Warning: We do not check for overflow *
//* *
//* Parameters *
//* item -- item to put in the stack *
//********************************************************/
//inline void stack::push(const int item)
//{
//	data[count] = item;
//	++count;
//}
///********************************************************
//* stack::pop -- get an item off the stack *
//* *
//Page 205
//Example 13-2. stack_c/stack_c.cc (Continued)
//* Warning: We do not check for stack underflow *
//* *
//* Returns *
//* the top item from the stack *
//********************************************************/
//inline int stack::pop(void)
//{
//	// Stack goes down by one
//	--count;
//	// Then we return the top value
//	return (data[count]);
//}
//// A short routine to test the stack
//int main(int argc, char *argv[])
//{
//	stack a_stack; // Stack we want to use
//	a_stack.init();
//	// Push three values on the stack
//	a_stack.push(1);
//	a_stack.push(2);
//	a_stack.push(3);
//	// Pop the items from the stack
//	cout << "Expect a 3 ->" << a_stack.pop() << '\n';
//	cout << "Expect a 2 ->" << a_stack.pop() << '\n';
//	cout << "Expect a 1 ->" << a_stack.pop() << '\n';
//	return (0);
//}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Example 13 - 1 - END <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

//>>>>>>>>>>>>Constructor :  ? 
//class stack {
//	// ...
//public:
//	// Initialize the stack
//	stack(void);
//	// ...
//};
//inline stack::stack(void)
//{
//	count = 0; // Zero the stack
//}

//IMPROVEMENT!!!!
//main()
//{
//	stack a_stack; // Stack we want to use
//	a_stack.init();

//	you can just write :
//	main()
//	{
//		stack a_stack; // Stack we want to use
//		// Use the stack

//>>>>>>>> Destructor : (~) 
//~ClassName

//stack:: ~stack(void)
//if (count != 0)
//cerr << "Error: Destroying a nonempty stack\n";
//}

//>>>>>>>>>>Parameterized Constructors

//class person {
//public:
//	char name[80]; // Name of the person
//	char phone[80]; // Person's phone number
//public:
//	person(const char i_name[], const char i_phone[]);
//	// ... rest of class
//};

//person::person(const char i_name[], const char i_phone[])
//{
//	strcpy(name, i_name);
//	strcpy(phone, i_phone);
//}
// USage : 
//main ()
//{
//person sam("Sam Jones", "555-1234");

//>>>>>>>>>>>>>>>< OVerload :
//
//class person {
//	// ... rest of the class
//public:
//	person(const char i_name[]);
//};
//Page 208
//person::person(const char i_name[])
//{
//	strcpy(name, i_name);
//	strcpy(phone, 'No Phone");
//}
//

///!\ WARNING : person unnamed_source; // Illegal

//>>>>>>>>>>>>>>>>><< Parameterized Destructors : DO NOT EXIST !!!!

// >>>>>>>>>>>>>>>> Copy Constructor
// 
//stack::stack(const stack& old_stack)
//{
//	int i; // Index used to copy the data
//	for (i = 0; i < old_stack.count; ++i)
//		data[i] = old_stack.data[i];
//}
//count = old_stack.count;

//USAGE : 
//stack a_stack; // A simple stack
//a_stack.push(l); // Put a couple of elements on the stack
//a_stack.push(2);
//>>>>>>>>> stack b_stack(a_stack); // Create a copy of the stack 
//
//>>>>>>>>>>>>>>>>>>< Automatically Generated Member Functions ? 

////  class::class(const class &old_class )  //copy constructor
//call_type first_var;
//// Call copy constructor to
//// make duplicate of first_var
//class_type second_var(first_var);
//class : : ~class()


//class class::operator = (const class &old_class)

//>>>>>>>>>>>>>>>>>>>>>> Shortcuts ?
//class stack {
//public:
//	// .... rest of class
//	// Push an item on the stack
//	void push(const int item);
//};
//inline void stack::push(const int item)
//{
//	data[count] = item;
//	++count;
//}
//can be written as :
//class stack {
//public:
//	// .... rest of class
//	// Push an item on the stack
//	void push(const int item)
//		data[count] = item;
//	++count;
//}
//};

	//@TODO :  Programming exercises?


	/*-----------------------------------------------------------------/
	/			CHAP 14: More on Classes							   /
	/-----------------------------------------------------------------*/

	//>>>>>>>>>>>>>>>>< FRIENDS :  ?

	//@TODO :  Programming exercises?

	/*-----------------------------------------------------------------/
	/	CHAP 15: Simple Pointers |  Based on 2009 edition. pag. 388	   /
	/-----------------------------------------------------------------*/
	
	//int thing; 			// Define "thing" (see Figure 15-2A)
	//int *thing_ptr; 		// Define "pointer to a thing" (see Figur
	//&thing : a pointer , objet in the memory
	
	//* :  Dereference (given a pointer, get the thing referenced)
	//& : Address of (given a thing, point to it)
	
	//A pointer to thing. thing is an object. The & (address of) operator gets
	//the address of an object (a pointer), so &thing is a pointer. For example:
	
	//thing_ptr = &thing; // Point to the thing
	// (See Figure 15-2A)
	//*thing_ptr = 5; // Set "thing" to 5
	// (See Figure 15-2B)
	

	//@TODO :  Programming exercises?

		/********************************************************
		 ********************************************************
		 ********		Based on 2009 edition. Pag. 422	  *******
		 ******** Part IV.Advanced programmingand concepts*******
		 ******** 										  *******
		 ********************************************************
		 ********************************************************/


	/*-----------------------------------------------------------------/
	/			CHAP 16: File Input/Output							   /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	
	/*-----------------------------------------------------------------/
	/			CHAP 17: Debugging and Optimization					   /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			CHAP 18: Operator Overloading						   /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			CHAP 19: Floating Point							      /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			CHAP 20: Advanced Pointers							   /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			CHAP 21: Advanced C lasses							   /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?


		/********************************************************
		 ********************************************************
		 ********										  *******
		 ********    V. Other languagesand features		  *******
		 ******** 										  *******
		 ********************************************************
		 ********************************************************/




	/*-----------------------------------------------------------------/
	/			Chapter 22. Exceptions							   	   /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			Chapter 23. Modular Programming					       /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			Chapter 24. Templates						           /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			Chapter 25. Standard Template Library				   /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			Chapter 26. Program Design						       /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			Chapter 27. Putting It All Together				   /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			Chapter 28. From C to C ++					           /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			Chapter 29. C ++'s Dustier C orners				   /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	/*-----------------------------------------------------------------/
	/			Chapter 30. Programming Adages				           /
	/-----------------------------------------------------------------*/
	//@TODO :  Programming exercises?
	
		/********************************************************
		 ********************************************************
		 ********										  *******
		 ********		Part VI: Appendixes	  			  *******
		 ******** 										  *******
		 ********************************************************
		 ********************************************************/
	



















