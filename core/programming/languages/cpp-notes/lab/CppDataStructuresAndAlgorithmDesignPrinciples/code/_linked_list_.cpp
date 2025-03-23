/*
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
*/

#include <iostream>
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

- BA: Base Address
- sizeof(type): Memory required for single element
*/

/*
Linked List (singly, doubly) - Core Concepts
- Each element is a NODE : data + pointer to the next element/Node*
- The 1st Node is called a HEAD
-------------    ------------   ------------    -----------
| data |next| -> |data |next|-> |data |next|-> |data |NULL|
-------------    ------------   ------------    -----------
HEAD                                            TAIL

- LL: Not contigous memory
- LL: can be allocated in random location of the memory (Not linear)
*/

// code
struct Node
{
    int data;
    Node *next;
};

class linked_list
{
private:
    Node *head, *tail;
public:
    // constructor
    linked_list()
    {
        head = NULL;
        tail = NULL;
    }

    // destructor
    ~linked_list()
    {
        Node* current = head;
        Node* next;

        while(current != nullptr){
            next = current->next; // Save the next node
            delete current; // Delete the current node <=> (free(current) in C-Style)
            current = next; // Move to the next node
        }
    }

    //setter add node
    void add_node(int n)
    {
        // creating a new node object
        // new:  create a new object and allocate the memory for the current data type
        Node* tmp = new Node; // C-Style: Node* tmp=(Node*)malloc(sizeof(Node*));
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

    void del_node(int n){
        // Do nothing
    }

    void display_list(){
        Node* temp = head;
        while (temp != NULL) {
            std::cout << temp->data << "->";
            temp = temp->next; // transverse
        }
        std::cout << "NULL" << std::endl;
    }
};


int main()
{

    linked_list ll;
    ll.add_node(1);
    ll.add_node(2);
    // display ll
    ll.display_list();

    return 0;
}

