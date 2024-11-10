/*
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
*/

#include <iostream>
#include "interface.h"
using namespace std;


//////////////////////////////////
//////// 1. Linked List //////////
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

/*
Linked List structure
Each element is a NODE : data + pointer to the next element/Node*
The 1st Node is called a HEAD
-------------    ------------   ------------    -----------
| data |next| -> |data |next|-> |data |next|-> |data |NULL|
-------------    ------------   ------------    -----------
HEAD                                            TAIL

/!\ LL : Not contigous memory
/!\ LL  : can be allocated in random location of the memory (Not linear)
*/

//code
struct node
{
    int data;
    node *next;
};

class linked_list
{
private:
    node *head, *tail;
public:
    //constructor
    linked_list()
    {
        head = NULL;
        tail = NULL;
    }
    //setter add node
    void add_node(int n)
    {
        //creating a new node object
        // new :  create a new object and allocate the memory for the current data type
        node *tmp = new node;
        // adding a new element
        tmp->data = n;
        tmp->next = NULL;

        if(head == NULL)
        {
            head = tmp;
            tail = tmp;
        }
        else
        {
            tail->next = tmp;
            tail = tail->next;
        }
    }
};



int main()
{

    linked_list a;
    a.add_node(1);
    a.add_node(2);

    return 0;
}

