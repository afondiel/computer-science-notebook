int exists(int ints[], int size_ints, int k)
{
    int i;
    int found = 0; // can be a boolean variables : true/false
    for (i = 0; i < size_ints; i++)
    {
        if(ints[i] == k )
        {
            printf("data found at index %d\n", i);
            found = 1;
            break;
        }
    }
    if(found == 0){
        printf("data not found\n");
    }

    return found;
}
