/*
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
*/

#include <iostream>
#include <forward_list>
#include "interface.h"
using namespace std;

//////////////////////////////////
//////// 4. Forward list       //////////
//////////////////////////////////

/*
Just like linked list but with performance optimization
*/

//structure
//definition
//forward_list<type> fwd_list = {1, 2, 3};
//declaration
//forward_list<type> fwd_list

int main()
{
    cout << "### std::forward list\\./ ### " << endl;

    forward_list<int> fwd_list = {1, 2, 3};
    //to add an element into the list
    fwd_list.push_front(0);     //list becomes  {0, 1, 2, 3}
    auto lt = fwd_list.begin();
    fwd_list.insert_after(lt, 5);    // list becomes {0, 5, 1, 2, 3}
    fwd_list.insert_after(lt, 6);    //list becomes {0, 6, 5, 1, 2, 3}
    fwd_list.pop_front();   //list becomes {6, 5, 1, 2, 3}
    fwd_list.erase_after(lt);   //list becomes {6, 1, 2, 3}
  //  fwd_list.erase_after(lt, fwd_list.end());   //list becomes {6}

    for(int&c : fwd_list)
        cout <<"fwd_list : " <<c<< endl;

    return 0;
}


