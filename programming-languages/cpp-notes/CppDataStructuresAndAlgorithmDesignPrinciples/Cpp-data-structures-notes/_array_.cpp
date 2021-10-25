/*
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
*/

#include <iostream>
#include <array>
#include "interface.h"
using namespace std;


//////////////////////////////////
//////// 2. Array       //////////
//////////////////////////////////

/*
contagious memory structure
-----------------------------------------------------------------
| data[0] |     data[1]     |      data[2]      |      data[3]  |
-----------------------------------------------------------------
    BA     BA+1*sizeof(type) BA+2*sizeof(type) BA+3*sizeof(type)
BA : Base Address
sizeof(type) : Memory required for single element
*/

/*** C-style Array

 - Work in C++ but are not commonly used because of LIMITATIONS
 -> Memory allocations and deallocations handled manually (failure to allocate and memory leak(adr not found...)
 -> The operator[] function doesn't check whether the size of the argument is LARGER than sizeof of any array
        ->> which may lead to a segmentation FAULT or Memory corruption if used INCORRECTLY
 -> 'nested'(container) synthax gets too complicated and can lead to unreadable data
 -> deep copying has to be implemented manualy

 /!\ SOLUTION : STD ARRAY !!!

***/

//declaration prototype
/*
 * std::array<datatype, array size> array;
 * array functions :
 - list::begin()
 - list::end()
 size
 front
 back
 empty
 push_front
 push_back
 pop_front
 pop_back
 reverse
 sort
 erase
 merge
 swap


 */


 // LIMITATION

 // IMPOSSIBLE to change the size of std::array (size must be constant at compile time and fixed)
 // in the runtime or DYNAMICALY thus No Malloc function allowed
 // No custom allocation it will always use stack memory It will always use stack memory

 // SOLUTION :
 // !!!!Vector!!!
 void print_array(const array<int, 5>&std_array)
 {

     for(int i = 0; i < std_array.size(); i++)
     {
         cout << "Array[" << i <<"] :" << std_array[i] << endl;
     }
 }


int main()
{
    cout << "### std::array \\./ ### " << endl;

    array<int, 5> std_array ={0, 1, 2, 3, 4};
    cout << std_array[2] << endl;

    cout << "The lenght of array is : " << std_array.size() << endl;

    // print array
    print_array(std_array);

    return 0;
}
