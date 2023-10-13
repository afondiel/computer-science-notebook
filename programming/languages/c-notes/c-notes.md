# C - Notes 

## Agenda

## Overview

- Created by richie & in 19
- Programming paradigm
- Design patterns
- Robust code recipe : 
	=> Safe/Secure - Maintenable/flexible - reliable - portable
	=> To prevent : 
		=> user errors, when invalid input is passed to the program.
		=> exhaustions, when the program tries to acquire shared resources.
		=> internal errors, due to bugs (e.g. dangling pointers).
## Applications


## Tools & frameworks

- Compiler 
- Assembler 
- Linker


## The C Programming Language

**HISTORY**

- Before C were : FORTRAN, COBOL, PASCAL, B....
- created in 1970 by Denis Richie => Goal Writing OS
- 1980 Bjarne Stroustrup created C++
- the JAVA was born from C++ to handle bugs.

**HOW C Works?**

-  Programs are divided : `Data + instrucions`

- Compiler => REVOLUTION

```
C LANGUAGE => COMPILER => Assembler => LINKER => Machine Language
```

### Data types


the size of some data type depends on the target/platform (windows, iOS, linux, mcu, raspberryPi...) to get the size => sizeof(type)
basic type 
- bool : 1 bit
- char : 1 byte
- int : 2/4 bytes (depends on the target)
- long : 4/8 bytes 
- long int : 8 bytes (depends on the target)

- float : 4/8 bytes 
- double : 8 bytes 
- long double : 8 bytes 

derived containers
- Array : numerical data type 
- str_array: string data type 
- pointer 
- functions 

advanced types/user-defined : 
- struct
- union
- enum
- volatile
- static...

=> Variables
=> Conditions
=> Functions 
=> Modular programming
=> Object based programming

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

=> Syntax error (compilation) 
=> segmentation error (runtime)
=> Overflow (stack and heap)
	=> buffer
	=> integer
=> string format (%d, %x, %n ...)
=> NULL pointer
=> divide by zero
=> Dangling pointer
=> barfing garbage
=> memory leaks
=> stackover flow (int  <= float)

### Solution  : check, check ....check
- Assortion : detect constraints
- Array indexing : whenever possible
- embedding magic numbers  : within objects, to check their type quickly.
- results : system call, library function  ...


## Hello World! 

```c
main( ) {
        printf("hello, world");
}
```


## References

Wikipedia:
- [C Programming](https://en.wikibooks.org/wiki/Category:Book:C_Programming) 


- https://github.com/afondiel/research-notes/tree/master/books/computer-science



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


