#include <cstdio>
#include <cstdlib>

int main() 
{
    const int N = 1000;
    int numbers[N];
    int mynumber = 42;
    int found_loc = -1;

    for (int i = 0; i < N; i++)
        numbers[i] = i;

    #pragma acc parallel loop copyin(numbers[0:N], mynumber) copy(found_loc)
    for (int i = 0; i < N; i++)
    {
        if (numbers[i] == mynumber)
        {
            #pragma acc atomic write
            found_loc = i;
        }
    }

    if (found_loc == -1) printf("Number not found \n");
    else printf("Number found at location %d \n", found_loc);
    
    return 0;
}