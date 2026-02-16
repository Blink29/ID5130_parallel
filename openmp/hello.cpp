#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    // Print OpenMP version and number of available threads
    cout << "Testing OpenMP..." << endl;
    cout << "Maximum threads available: " << omp_get_max_threads() << endl;
    
    // Parallel region - each thread prints its ID
    #pragma omp parallel num_threads(10) 
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        #pragma omp critical
        {
            cout << "Hello from thread " << thread_id << " of " << num_threads << " threads!" << endl;
        }
        // printf("Hello from thread %d of %d threads!\n", thread_id, num_threads);
    }
    
    cout << "\nOpenMP is working correctly!" << endl;
    
    return 0;
}
