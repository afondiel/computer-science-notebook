/*
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
*/

#include <iostream>
#include <list>
#include "interface.h"
using namespace std;

//////////////////////////////////
//////// 5. List       //////////
//////////////////////////////////

/*
Just like forward list but with on limitation adding element back forward and more */

//structure
//definition
//_list<type> _list = {1, 2, 3};
//declaration
//_list<type> _list

int main()
{
    cout << "### std::forward list\\./ ### " << endl;

    list<int> _list = {1, 2, 3};
    //to add an element into the list
    //_list.push_front(0);     //list becomes  {0, 1, 2, 3}
    _list.push_back(0);     //list becomes  {0, 1, 2, 3}
    //auto lt =_list.begin();
    _list.insert_after(next(_list.begin()), 5);    // list becomes {0, 5, 1, 2, 3}
    _list.insert_after(_list.begin(), 6);    //list becomes {0, 6, 5, 1, 2, 3}
    _list.pop_back();   //list becomes {6, 5, 1, 2, 3}
    _list.erase_after(lt);   //list becomes {6, 1, 2, 3}
  //  fwd_list.erase_after(lt, fwd_list.end());   //list becomes {6}

    for(int&c : _list)
        cout <<"std_list : " <<c<< endl;

    return 0;
}



