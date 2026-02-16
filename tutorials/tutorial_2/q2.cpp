#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        for (int i = 0; i < num_threads; i++) {
            if (i == id) {
                printf("Hello from thread %d out of %d\n", id, num_threads);
            }
            #pragma omp barrier
        }
    }
    return 0;
}