/*=================================================================================
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
==================================================================================*/

#include <iostream>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

////////////////////////////
//////// 10. Heap //////////
////////////////////////////

/*

A Heap is a special *Tree-based* data structure in which the tree is a complete binary tree.
Generally, Heaps can be of two types:
 - Max-Heap : father node(the first one on the top) has the Max value
 - Min-Heap : father node(the first one on the top) has the min value

 Heap is used to handle binary tree (create, delete, arrange ... )
src : https://www.geeksforgeeks.org/heap-data-structure/

 - High level operator/algorithm than vector, set ... ?
*/


/*---- structure -----*/
//declaration
// - vector<type> _hT

//definition
// - vector<type> _hT = ?

// Operations/functions ----------------------x------------------------------ Time complexity
/*
- insert()
- deletion()  : the 1st element is deleted first then we swap w/ the last one
- peek()
- Extract-Max/Min

*/

/*
Heap Data Structure Applications :
    - Heap is used while implementing a priority queue.
    - Dijkstra's Algorithm
    - Heap Sort
*/


/* --------------------- code -----------------------*/
//// Max-Heap data structure in C++
// Src : https://www.programiz.com/dsa/heap-data-structure

void _swap(int *a, int *b)
{
    int temp;
    temp = *b;
    *b = *a;
    *a = temp;
}

void heapify(vector<int> &hT, int i)
{
    int _size = hT.size();
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    if (l < _size && hT[l] > hT[largest])
        largest = l;
    if (r < _size && hT[r] > hT[largest])
        largest = r;

    if (largest != i)
    {
        _swap(&hT[i], &hT[largest]);
        heapify(hT, largest);
    }
}

void _insert(vector<int> &hT, int newNum)
{
    int _size = hT.size();
    if (_size == 0)
    {
        hT.push_back(newNum);
    }
    else
    {
        hT.push_back(newNum);
        for (int i = _size / 2 - 1; i >= 0; i--)
        {
            heapify(hT, i);
        }
    }
}

void deleteNode(vector<int> &hT, int num)
{
    int _size = hT.size();
    int i;
    for (i = 0; i < _size; i++)
    {
        if (num == hT[i])
            break;
    }
    _swap(&hT[i], &hT[_size - 1]);

    hT.pop_back();
    for (int i = _size / 2 - 1; i >= 0; i--)
    {
        heapify(hT, i);
    }
}

void printArray(vector<int> &hT)
{
    for (int i = 0; i < hT.size(); ++i)
        cout << hT[i] << " ";
    cout << "\n";
}

int main()
{
    cout << "HEap Tree ";
    vector<int> heapTree;

    _insert(heapTree, 3);
    _insert(heapTree, 4);
    _insert(heapTree, 9);
    _insert(heapTree, 5);
    _insert(heapTree, 2);

    cout << "Max-Heap array: ";
    printArray(heapTree);

    deleteNode(heapTree, 4);

    cout << "After deleting an element: ";

    printArray(heapTree);

    return 0;

}

