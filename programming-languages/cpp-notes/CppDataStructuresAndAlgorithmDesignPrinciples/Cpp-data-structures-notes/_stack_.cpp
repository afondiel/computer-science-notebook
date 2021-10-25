/*=================================================================================
* From the Book : C++ Data Structures and Algorithm Design Principles
* By :  John Carey
* Src FREE course: https://www.youtube.com/watch?v=lF1UOAvkPnE&t=146s
==================================================================================*/

#include <iostream>
#include <stack>
#include <bits/stdc++.h>

using namespace std;

//////////////////////////////////
//////// 7. Stack       //////////
//////////////////////////////////

/*
data structure based on  : LIFO = LAST IN FIRST OUT
for accessing and processing the data
*/

/*
|   last   |
----------
|          |
------------
|          |
----------
|   first  |
------------
*/

/*---- structure-----*/
//=== declaration ====
//stack<type> _stack
//=== definition ======
//stack<type> _stack = {1, 2, 3};

// functions ----------------------x------------------------------ Time complexity
/*
- empty() :  returns whether the stack is empty                     ------------- O(1)
- size() : returns the size of the stack                            ------------- O(1)
- top() : returns a reference to the top most element of the stack  ------------- O(1)
- push(g) : adds the element 'g' at the top of the stack            ------------- O(1)
- pop() : deletes the top most element of the stack                 ------------- O(1)
-
*/


void ShowStackData(stack <int> s)
{
    while(!s.empty())
    {
        cout << '\t' << s.top();
        s.pop();
    }
    cout << '\n';

}

int main()
{
    cout << "### std::stack\\./ ### " << endl;

    stack<int> _stack;
    // Add element in the stack
    _stack.push(1);
    _stack.push(3);
    _stack.push(2);
    _stack.push(5);
    _stack.push(1);

    cout <<"The _stack is : ";
    ShowStackData(_stack);

    cout << "\ns.size() :" << _stack.size();
    cout << "\ns.top() :" << _stack.top();

    //removing element from the stack
    cout << "\ns.pop() : ";
    _stack.pop();  //remove the first element
    ShowStackData(_stack);

    return 0;
}



