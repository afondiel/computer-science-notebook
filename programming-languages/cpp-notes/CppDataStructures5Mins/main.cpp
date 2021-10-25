#include <iostream>
#include <array>
// 5 Common data structure in C++
#include <vector>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>

using namespace std;


int main()
{
    //cout << "Hello world!" << endl;

    /***
    * Vectors are JUST LIKE THE "STD::ARRAY" except the size is not constant
    * and can change dynamically or during the run time.
    ***/

    //Synthax :
    //=> std::vector<datatype> vector_name;
    //=> SAME FUNCTIONS as the std::array
    //Most common Methods :
    //=> push_back()                => runtime : constant O(1)
    //=> [] : bracket operators     => runtime : constant O(1)
    //=> size()                     => runtime :

    /* VECTOR */
    /*vector<int> v;              // v = {}
    cout << v.size() << endl;     // output 0
    v.push_back(20);            // v = {20}
    v.push_back(10);            // v = {20,10}
    v.push_back(10);            // v = {20, 10, 10}
    cout << v[1] << endl;       // outputs 10
    cout << v.size() << endl;   //outputs 3
    */

    /***
    * SET are just like the "STD::VECTOR" except the elements are sorted order and
    * doesnot contain duplicated elements
    ***/

    //Synthax :
    //=> std::set<datatype> set_name;
    //=> SAME FUNCTIONS as the std::vector
    //Most common Methods :
    //=> insert()                   => runtime : O(logn)
    //=> find()                     => runtime : O(logn)
    //=> size()                     => runtime : constant O(1)
    /* SET */

    /*set<int> s;                     // s = {}
    cout << s.size() << endl;       // output 0
    s.insert(20);                   // s = {20}
    s.insert(10);                   // s = {10, 20}
    s.insert(10);                   // s = {10, 20}
    //auto it = s.find (10);          //it is a iterator that points to 10
    auto it = s.find (20);          //it is a iterator that points to 20
    cout << (it != s.end() ? "FOUND" : "" )<< endl;       // outputs FOUND
    cout << s.size() << endl;        //outputs 2*/



    /***
    * UNORDERED_SET are just like the "STD::SET" except the elements are COULD BE in any ORDER and
    * doesnot contain duplicated elements
    ***/

    //Synthax :
    //=> std::unordered_set<datatype> unordered_set_name;
    //=> SAME FUNCTIONS as the std::set
    //Most common Methods :
    //=> insert()                   => runtime : constant O(1)
    //=> find()                     => runtime : constant O(1)
    //=> size()                     => runtime : constant O(1)
    /* UNORDERED_SET*/


   /* unordered_set<int> u_s;                     // u_s = {}
    cout << u_s.size() << endl;       // output 0
    u_s.insert(20);                   // u_s = {20}
    u_s.insert(10);                   // u_s = {10, 20} this could be in ANY order
    // Exemple when using a for loop :
    //for(auto e : u_s){ ...}  //range loop
    //for(auto it = u_s.begin(); it! = u_s.end(); ++it){ ... }  //regular loop
    u_s.insert(10);                   // u_s = {10, 20}
    //auto it = s.find (10);          //it is a iterator that points to 10
    auto it = s.find (20);          //it is a iterator that points to 20
    cout << (it != u_s.end() ? "FOUND" : "" )<< endl;       // outputs FOUND
    cout << u_s.size() << endl;        //outputs 2
    */


    /*** MAP **** // Like a Dictionnary in PYTHON!!!! // OR HASH TABLE in PERL!!!
    * are just like the "STD::SET" except instead of storing an element or a value,
    * it stores a <key:value> pair
    ***/

    //Synthax :
    //=> std::map<datatype> map_name;
    //=> SAME FUNCTIONS as the std::map
    //Most common Methods :
    //=> insert()                   => runtime : O(logn) : insert using "make_pair"
    //=> find()                     => runtime : O(logn) : return "pair"
    //=> [] : bracket operators     => runtime : O(logn) : return "ref to value"
    //=> size()                     => runtime : constant O(1)
    /* UNORDERED_SET*/
/*
    map<int, int> m;                // m = {}
    cout << m.size() << endl;       // output 0
    m.insert(make_pair(20, 1));     // m = {(20,1)}
    m.insert(make_pair(10, 1));     // m = {(10,1) , (20,1) }
    m[10]++;                         // m = {(10,2) , (20,1) }
    auto it = m.find (10);          //it is an iterator that points to (10,2)
    cout << (it != m.end() ? it->second : 0 )<< endl;       // outputs 2
    auto it2 = m.find (20);          //it is an iterator that points to (20,1)
    cout << (it2 != m.end() ? it->first : 0 )<< endl;       // outputs 20
    cout << m.size() << endl;        //outputs 2

    */

/*** UNORDERED MAP **** // Like a Dictionnary in PYTHON!!!! // OR HASH TABLE in PERL!!!
    * are just like the "STD::MAP" except instead of storing an element or a value,
    * it stores a <key:value> pair IN ANY ORDER
    ***/

    //Synthax :
    //=> std::unordered_map<datatype> map_name;
    //=> SAME FUNCTIONS as the std::map
    //Most common Methods :
    //=> insert()                   => runtime : O(logn) : insert using "make_pair"
    //=> find()                     => runtime : O(logn) : return "pair"
    //=> [] : bracket operators     => runtime : O(logn) : return "ref to value"
    //=> size()                     => runtime : constant O(1)
    /* UNORDERED_SET*/

    unordered_map<int, int> m;                // m = {}
    cout << m.size() << endl;       // output 0
    m.insert(make_pair(20, 1));     // m = {(20,1)}
    m.insert(make_pair(10, 1));     // m = {(10,1) , (20,1) }
    m[10]++;                         // m = {(10,2) , (20,1) }
    auto it = m.find (10);          //it is an iterator that points to (10,2)
    cout << (it != m.end() ? it->second : 0 )<< endl;       // outputs 2
    auto it2 = m.find (20);          //it is an iterator that points to (20,1)
    cout << (it2 != m.end() ? it->first : 0 )<< endl;       // outputs 20
    cout << m.size() << endl;        //outputs 2*/


    return 0;
}
