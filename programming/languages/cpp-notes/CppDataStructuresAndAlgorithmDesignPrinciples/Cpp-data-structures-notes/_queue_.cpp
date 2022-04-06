/*=================================================================================
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
==================================================================================*/

#include <iostream>
#include <queue>
#include <bits/stdc++.h>

using namespace std;

//////////////////////////////////
//////// 8. Queue       //////////
//////////////////////////////////

/*
data structure based on  : FIFO = FIRST IN FIRST OUT
for accessing and processing the data

 -> Removing from the front
 -> Adding from the back

*/

/*
|   last/TAIL   |          |          |   first/HEAD  |
*/

/*---- structure-----*/
//=== declaration ====
//queue<type> _queue
//=== definition ======
//queue<type> _queue = {1, 2, 3};

// functions ----------------------x------------------------------ Time complexity
/*
- empty() :  returns whether the queue is empty                     ------------- O(1)
- size() : returns the size of the queue                            ------------- O(1)
- front() : returns a reference to the top most element of the queue  ------------- O(1)
- push(g) : adds the element 'g' at the top of the queue            ------------- O(1)
- pop() : deletes the top most element of the queue                 ------------- O(1)
-
*/


void ShowqueueData(queue <int> q)
{
    while(!q.empty())
    {
        cout << '\t' << q.front();
        q.pop();
    }
    cout << '\n';

}

int main()
{
    cout << "### std::queue\\./ ### " << endl;

    queue<int> _queue;
    // Add element in the queue from the BACK of the queue
    _queue.push(1);  // queue becomes {1}
    _queue.push(3);  // queue becomes {1, 3}
    _queue.push(2);   // queue becomes {1, 3, 2}

    cout << "The size of the queue is :" << _queue.size() << endl;

    //removing element from the FRONT of the queue (tail)
    _queue.pop();  //queue becomes {2, 3}
    _queue.push(4);  //queue becomes {2, 3, 4}
    ShowqueueData(_queue);

    return 0;
}



