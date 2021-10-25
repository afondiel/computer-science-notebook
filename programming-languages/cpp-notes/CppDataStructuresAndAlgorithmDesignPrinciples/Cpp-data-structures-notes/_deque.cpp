/*=================================================================================
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
==================================================================================*/

#include <iostream>
#include <deque>

using namespace std;

//////////////////////////////////
//////// 6. Deque       //////////
//////////////////////////////////

/*
 it's a double-ended queue used to store, maintain and handle the data
just like vector but used to expansion and contraction but more efficient
insert, deletion of elements are possible at the HEADS (opposite to Queue)

 * USAGE : pop and push BOTH from front and back
  /!\ It's also a low level layer thant Queue or stack
    /!\std::stack<int, std::deque<int> > s;
    /!\std::queue<double, std::list<double> > q;
  /!\
*/

//structure
//definition
//deque<type> _deque = {1, 2, 3};
//declaration
//deque<type> _deq

int main()
{
    cout << "### std::deque\\./ ### " << endl;

    deque<int> _deq = {1, 2, 3, 4, 5};
    _deq.push_front(0);                         //deq becomes  {0,1, 2, 3, 4, 5}
    _deq.push_back(6);                          //deq becomes  {0,1, 2, 3, 4, 5, 6}
    _deq.insert(_deq.begin() + 2, 10);          // deq becomes {0, 1, 10, 4, 5, 6}
    _deq.pop_back();                            //deq becomes  {0, 1, 10, 2, 3, 4, 5}
    _deq.pop_front();                           //deq becomes  {1, 10, 3, 4, 5, 6}
    _deq.erase(_deq.begin() + 1);                 //deq becomes  {1, 2, 3, 4, 5, 6}
    _deq.erase(_deq.begin() + 3, _deq.end());   //deq becomes {1, 2, 3}


    cout <<"std_deq : ";
    for(auto i: _deq)
        cout << i << " ";

    return 0;
}



