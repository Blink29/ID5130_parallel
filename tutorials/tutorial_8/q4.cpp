#include <cstdio>

// Version 1
void version1()
{
    int a[10];

    printf("\n===== VERSION 1 =====\n");
    #pragma acc kernels
    {
        printf("One \n");
        for (int i = 0; i < 10; ++i)
            a[i] = i;
        printf("Two: a[9] = %d \n", a[9]);
        for (int i = 0; i < 10; ++i)
            a[i] *= 2;
        printf("Three: a[9] = %d \n", a[9]);
    }
}

// Version 2
void version2()
{
    int a[10];

    printf("\n===== VERSION 2 =====\n");
    #pragma acc kernels
    {
        printf("One \n");
        for (int i = 0; i < 10; ++i)
        {
            a[i] = i;
            printf("first loop %d \n", i);
        }
        printf("Two: a[9] = %d \n", a[9]);
        for (int i = 0; i < 10; ++i)
        {
            a[i] *= 2;
            printf("second loop %d \n", i);
        }
        printf("Three: a[9] = %d \n", a[9]);
    }
}

// Version 3
void version3()
{
    int a[10];

    printf("\n===== VERSION 3 =====\n");
    #pragma acc kernels
    {
        printf("One \n");
        for (int i = 0; i < 10; ++i)
        {
            a[i] = i;
            printf("first loop %d \n", i);
        }
        printf("Two \n");
        for (int i = 0; i < 10; ++i)
        {
            a[i] *= 2;
            printf("second loop %d \n", i);
        }
        printf("Three \n");
    }
}

int main()
{
    version1();
    version2();
    version3();
    return 0;
}
