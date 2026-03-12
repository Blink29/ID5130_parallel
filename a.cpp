#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

int main() {
    int x = 10;
    #pragma omp parallel num_threads(4)
{
 int myrank = omp_get_thread_num();
 printf("\n x = %d on thread %d before init", x, myrank);
 x = myrank * 10;
 printf("\n x = %d on thread %d after init", x, myrank);
 }
 printf("\n x = %d after exit from parallel \n", x);
    return 0;
}
