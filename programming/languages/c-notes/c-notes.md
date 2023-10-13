# C - Notes 

## Overview

- Created by richie & in 19
- Programming paradigm
- Design patterns

## Applications

- Operating Systems (The UNIX, Windows and OSX kernels  ...)
- Embedded Systems
- GUI (Graphical User Interface)
- Google
- Design of a Compiler
- Mozilla Firefox and Thunderbird
- Gaming and animation
- MySQL

Src: [Most Useful Applications of C Programming Language 2023 - Interviewbit](https://www.interviewbit.com/blog/applications-of-c-programming-language/)

## Tools & frameworks

- Compiler 
- Assembler 
- Linker

[awesome-c: A curated list of awesome C frameworks, libraries and software.](https://github.com/uhub/awesome-c)

## The C Programming Language

**HISTORY**

- Before C were : FORTRAN, COBOL, PASCAL, B....
- created in 1970 by Denis Richie => Goal Writing OS
- 1980 Bjarne Stroustrup created C++
- the JAVA was born from C++ to handle bugs.

**HOW C Works?**

Programs are divided : `Data + instrucions`

- Compiler => REVOLUTION!!!

```
C LANGUAGE => COMPILER => Assembler => LINKER => Machine Language
```
**HOW TO LEARN C?**

- only way : write program + Debbuging
- Keep good programming style
- comments helps to organize thoughts even before writing CODE/PROGRAM
- Think portability

### Compliation Process

1. Pre-processing
2. Compilation
3. Assembling
4. Linkage

To compile a c program using command line:

Linux: 

```sh
gcc file.c -g –o file.o
```

Windows: 

```bat
gcc file.c -g –o file.exe
```

Options/flags
- g : enables debugging
- Wall : enables warning

### Data types

The size of some data type depends on the target/platform (windows, iOS, linux, mcu, raspberryPi...) 

To get the size => sizeof(type)

**Basic types** 

- bool : 1 bit
- char : 1 byte
- int : 2/4 bytes (depends on the target)
- long : 4/8 bytes 
- long int : 8 bytes (depends on the target)

- float : 4/8 bytes 
- double : 8 bytes 
- long double : 8 bytes 

**Derived containers**

- Array : numerical data type 
- str_array: string data type 
- pointer 
- functions 

**Advanced types/user-defined** 

- struct
- union
- enum
- volatile
- static...


### Variables
### Conditions
### Functions 

### Modular programming
### Object based programming

## Data structure & Algorithms (DSA)

 
### Time & Space complexity (Big O, Big Omega, Big Theta)

- time => speed
- space => Memory

For more insights check out the [data structures notes](../../data-structures/ds-notes.md).

**DSA Family**

- Accessing : O(n)
- Searching: O(logn)
- Inserting: O(n2) 
- Deleting: O(n2)


**DSA Lists**

- Linked Lists
- Stack
- Queue
- Sets
- Hash tables
- Trees
- Heaps and Priority Queues
- Graphs
- Sorting
- searching
- Numerical Methods
- Data Compression
- Data Encryption
- Graph Algorithms
- Geometric Algorithms


## Debugging & Error Handling

**Two common errors**

- Syntax error: occurs during the `compilation` 
- Segmentation error: occurs during the `runtime`


A Non-exhaustive List of errors in c programming:

- Overflow (stack and heap)
	- buffer
	- integer
- string format (%d, %x, %n ...)
- NULL pointer
- divide by zero
- Dangling pointer
- barfing garbage
- memory leaks
- stackover flow (int  <= float)

- user errors, when invalid input is passed to the program.
- exhaustions, when the program tries to acquire shared resources.
- internal errors, due to bugs (e.g. dangling pointers).

## Good Practice: check, check ... check

Robust code recipe: 

- Safe/Secure 
- Maintenable/flexible
- reliable
- portable
- Assortion : detect constraints
- Array indexing : whenever possible
- embedding magic numbers  : within objects, to check their type quickly.
- results : system call, library function  ...


## Hello World! 

Program to Display "Hello, World!"

```c
#include <stdio.h>
int main() {
   // printf() displays the string inside quotation
   printf("Hello, World!");
   return 0;
}
```
Output:

```
Hello, World!
``` 

## References

Wikipedia:
- [C Programming](https://en.wikibooks.org/wiki/Category:Book:C_Programming) 

WikiBooks:
- [C Programming](https://en.wikibooks.org/wiki/Category:Book:C_Programming) 


Courses & Tutorials:

Books:

- [C Programming Books](https://github.com/afondiel/cs-books/tree/main/computer-science/programming/C)


Debugging:

- [C Programming/Error handling](https://en.wikibooks.org/wiki/C_Programming/Error_handling)
- [Robust Design Techniques for C Programs](https://freetype.sourceforge.net/david/reliable-c.html)
- [Error Handling in C++ or: Why You Should Use Eithers in Favor of Exceptions and Error-codes :](https://hackernoon.com/error-handling-in-c-or-why-you-should-use-eithers-in-favor-of-exceptions-and-error-codes-f0640912eb45) 


Modern C++ best practices for exceptions and error handling:

- https://learn.microsoft.com/en-us/cpp/cpp/errors-and-exception-handling-modern-cpp?view=msvc-170
- https://learn.microsoft.com/en-us/cpp/cpp/errors-and-exception-handling-modern-cpp

Standards:
- [MISRA ](https://ldra.com/misra/)


