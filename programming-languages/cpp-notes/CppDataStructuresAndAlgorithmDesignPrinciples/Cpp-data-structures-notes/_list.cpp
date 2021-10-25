/*=================================================================================
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
==================================================================================*/

#include <iostream>
#include <list>

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
    cout << "### std::list\\./ ### " << endl;

    list<int> _list = {1, 2, 3, 4, 5};
    // Add an element at the end of the list
    _list.push_back(6);  //list becomes  {1, 2, 3, 4, 5, 6}

    /* dont need to create an iterator like in forward list to insert the next element*/
    //auto lt =_list.begin();
    _list.insert(next(_list.begin()), 0);    // list becomes {1, 0, 2, 3, 4, 5, 6}
    //_list.insert(_list.begin(), 6);        //list becomes {1, 0, 2, 3, 4, 5, 6}
    _list.insert(_list.end(), 7);            //list becomes {1, 0, 2, 3, 4, 5, 6, 7}
    /* delete the last element from the list*/
    _list.pop_back();       //list becomes {1, 0, 2, 3, 4, 5, 6}
    // Add an element at the begin of the list
    _list.push_front(-1);   //list becomes {-1, 1, 0, 2, 3, 4, 5, 6}

   // _list.erase_after(lt);   //list becomes {6, 1, 2, 3}
   //  fwd_list.erase_after(lt, fwd_list.end());   //list becomes {6}

    cout <<"std_list : ";
    for(auto i: _list)
        cout << i << " ";

    return 0;
}



