#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

void Hello() {
#ifdef _OPENMP
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
#else
    int my_rank = 0;
    int thread_count = 1;
#endif
    printf("Hello from thread %d out of %d\n", my_rank, thread_count);
}

int main(int args, char* argv[]) {
    int thread_count = 1;
    if (args == 2) {
        thread_count = strtol(argv[1], NULL, 10);
    }
    else {
        printf("A command line argument other than name of the executable is required...exiting the program..\n");
        return 1;
    }

    #pragma omp parallel num_threads(thread_count)
    {
        Hello();    
    }
    return 0;
}