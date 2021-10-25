#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
//#include <sys/resource.h>

using namespace std;


// cout << "Hello world!" << endl;


//Q1:C++
//Q2:C++
//Q3-C++ : Comments

    //This is a comment
    /*This is another comment*/

//Q4-C++ : Check the presence of a number in an array
//code

class Answer
{
public:
    //save CPU : Big(O) ?
    static bool exists(int ints[], int size_ints, int k)
    {
        int i;
        for (i = 0; i < size_ints; i++){
            if(ints[i] == k )
                return true;
        }
        return false;
    }

};



//Q5:C++
//Q6:C++
//Q7:C++
//Q8:C++
//Q9:C++
//Q10:C++
//Q11:C++
//Q12:C++

void Q12(void)
{
    int a = 5;
    int *p = &a;

    cout << "a:"<<a <<"&a:" << &a <<"p:" << p <<"*p:" << *p <<endl;
}

//Q13:C++
//Q14:C++
//Q15:C++

int a(){
    cout << "a";
    return 0;
}
int b(){
    cout << "b";
    return 0;
}

//Q16:C++

void Q16(void)
{
    char *str = "hello";

    cout << sizeof(str) << endl; // 8 on 64 bytes PC : it depends on systems architecture and compiler
}

//Q17-C++ : lambda expression

void Q17(void)
{
    int a = 5;
    auto f = [&]{ return a;}; //reference
    auto g = [=]{ return a;}; //value
    a = 6;
    int delta = f() - g();

    cout << delta << endl; //1 f() is a reference therefore it changes
}


//Q18 - C++ : Lambada expressiion

struct A{
    int value;
};
struct B: public A{
    B(){value = 0;}
};
struct C: public A{
    C(){value = 1;}
};
struct D: public B, public C{
    //cout << value;
};


//Q19:C++
//Q20:C++
//Q21:C++


// Main
int main()
{
    //Q4
    //int ints[] = {-9, 14, 37, 102};
    // call-1
    //Answer a;
    //cout << a.exists(ints, 4, 102) << endl; //1
    //cout << a.exists(ints, 4, 36) << endl; //1

    //call-2
    //cout << Answer::exists(ints, 4, 102) << endl; //1
    //cout << Answer::exists(ints, 4, 36) << endl; //0

    //Q12
    //Q12();

    //Q15
    if(a() & b())
        cout << "main"; // res :  ?

    //Q16
    //Q16();

    //Q18
//    D d;
//    std::cout << d.value; // Build fails ambiguity

    //Q17
    //Q17();

    return 0;
}
