#include <iostream>
#include <array>
#include <vector>
#include "interface.h"
using namespace std;

/***
JUST LIKE THE "STD::ARRAY" except the size is not constant and can change
dynamically or during the run time.
***/

//Synthaxe
//std::vector<datatype> array_name;
//SAME FUNCTIONS as the std::array

void print_all_vector(const vector<int> studentmarks)
{
    for(int i=0; i < studentmarks.size(); i++ ){
        cout << "studentmarks[" << i << "] = " << studentmarks[i] << endl;
    }
}

void print_first_element(const vector<int> studentmarks)
{
    cout << "First element of studentmarks is : " << studentmarks.front() << endl;
}

void print_last_element(const vector<int> studentmarks)
{
    cout << "The last element of studentmarks is : " << studentmarks.back() << endl;
}

void get_element_in_the_position_chosen(const vector<int> studentmarks, int position)
{
    cout << "The value of the element " << position << "is : " << studentmarks.at(position) << endl;
}


//main function
int main(void)
{
    //vector<int> marks;
    //vector<int> marks(10); //declares vector of size of 10
    //vector<int> marks(10, 5); //declares vector of size of 10 with each element value 5
    //print the size
    //cout << "size of thee vector marks: "  << marks.size() << endl;

    vector<int> studentmarks = {20, 30, 45, 60, 90};
    //add a new element dynamically during the runtime
    studentmarks.push_back(1);
    // add element in the begin of array
    studentmarks.insert(studentmarks.begin(), 2);

    cout << "length of array :" << studentmarks.size() << endl;

    //for(int i=0; i < studentmarks.size(); i++ ){
      //  cout << "studentmarks[" << i << "] = " << studentmarks[i] << endl;
    //}
    //print the vector
    print_all_vector(studentmarks);

    // print first element
    print_first_element(studentmarks);

    // last first element
    print_last_element(studentmarks);

    //add_new_element in the specific position
    get_element_in_the_position_chosen(studentmarks, 2);
    return 0;
}
