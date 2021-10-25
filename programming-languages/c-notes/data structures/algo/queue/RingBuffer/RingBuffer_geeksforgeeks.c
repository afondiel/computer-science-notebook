//-------------------------------------------------------------------//
//-------------------------------------------------------------------//
//-----------------------Ring Buffer---------------------------------//
//-------------------------------------------------------------------//
//-------------------------------------------------------------------//
//-------------------------------------------------------------------//

/*
 1. Ring/Circular buffer/queue is a linear data structure
 2. Based on FIFO (First In First Out
 3. Has 2 ends 'front' and 'rear'
 4. INSERTION from 'rear'
 5. DELETION From 'front'

-------------------------------------
| 0|20|50| 7 | 30 | 15 | 60 | 34 | 54|
-------------------------------------
 ^                                  ^
front                               rear
//<---- deletion---->
---------------------------------------
| x| x |50| 7 | 30 | 15 | 60 | 34 | 54|
---------------------------------------
        ^                           ^
      front                        rear
//<---- Insertion---->
----------------------------------------
| 24| 18 | 50 | 7 | 30 | 15 | 60 | 34|54|
----------------------------------------
      ^    ^
    rear front
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

void _insert(void);
void _delete(void);
void _display(void);
void _menu(void);


//Global vars
#define MAX 5
int q[MAX];
int front=-1, rear=-1;

int main(void)
{
    int ch;
    bool loop = true;
    printf("//-------circular Queue operation-------------//\n");
   // _menu();

    while(loop)
    {
        _menu();
        printf("Enter your choice : ");
        scanf("%d", &ch);
        switch(ch)
        {
        case 1 : _insert();
            break;
        case 2 : _delete();
            break;
        case 3 : _display();
            break;
        case 4 :
            loop = false;
            printf("See you soon! \n");
            break;
        default : printf("Invalid Option, please try again\n");
            break;
        }

    }

    return 0;
}

void _insert(void)
{
    //--------------------------------//
    // time complexity O(1) no loop   //
    //------------------------------- //
    int x;
    //Empty queue : rear==front==-1
    if(((front == 0)&& (rear == MAX - 1)) || (rear + 1 == front))
    {
        printf("Queue is overflow\n");
    }
    else
    {
        printf("Enter element to be insert : \n");
        scanf("%d", &x);
        if(rear == -1){
            front = 0;
            rear = 0;
        }
        else if (rear = MAX-1)
        {
            rear = 0;
        }
        else
        {
            rear++;
        }
        //add new element into the queue
        q[rear] = x;
    }
}

void _delete()
{
//Delete while rear!= front
    int a;
    if(front == -1)
    {
        printf("Queue is underflow\n");
    }
    else
    {
        a  = q[front];
        if(front == rear)
        {
            front = -1;
            rear = -1;
        }
        else if(front == MAX -1){
            front = 0;
        }
        else
        {
            front++;
        }
        printf("Delete element is %d\n", a);
    }

}

void _display()
{
    int i;
    if(((front == 0)&& (rear == MAX - 1)) || (rear + 1 == front))
    {
        printf("Queue is overflow\n");
    }
    else
    {
        for(i=front; i<rear; i++)
        {
            printf("Queue : ", q[i]);
        }
    }
}

void _menu(void)
{
    printf("1. Insert\n2. Delete\n3. Display\n4. Exit\n");

}

