
/*=================================================================================
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
==================================================================================*/

#include <iostream>
#include <queue>
#include <bits/stdc++.h>

using namespace std;

/////////////////////////////////////
//////// 9. Priority Queue //////////
/////////////////////////////////////

/*
It's just like the Queue but PQ aranges element in a NON increasing order
{5,7,10} => becomes {10, 7, 5}

 /!\ PQ is based on vector

 -> Removing from the front
 -> Adding from the back

 */

/*
|   last/TAIL   |          |          |   first/HEAD  |
*/

/*---- structure-----*/
//=== declaration ====
//priority_queue<type> _p_queue
//=== definition ======
//priority_queue<type> _pqueue = {1, 2, 3};

// functions ----------------------x------------------------------ Time complexity
/*
- empty() :  returns whether the queue is empty                     ------------- O(1)
- size() : returns the size of the queue                            ------------- O(1)
- front() : returns a reference to the top most element of the queue  ------------- O(1)
- push(g) : adds the element 'g' at the top of the queue            ------------- O(1)
- pop() : deletes the top most element of the queue                 ------------- O(1)
-
*/


void ShowPqueueData(priority_queue <int> pq)
{
    while(!pq.empty())
    {
        cout << '\t' << pq.top();
        //removing element
        pq.pop();
    }
    cout << '\n';
}

int main()
{
    cout << "### std::priority_queue\\./ ### " << endl;

    priority_queue<int> _pqueue;
    // Add element in the _pqueue from the BACK of the _pqueue
    _pqueue.push(1);  // _pqueue becomes {1}
    _pqueue.push(3);  // _pqueue becomes {3, 1}
    _pqueue.push(2);   // _pqueue becomes {3, 2, 1}

    cout << "The size of the priority_queue is :" << _pqueue.size() << endl;
    cout << "The top of the priority_queue is :" << _pqueue.top() << endl;

    ShowPqueueData(_pqueue);

    return 0;
}



