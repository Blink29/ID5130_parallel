#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    int num_steps = 1000000000;
    double steps, pi, sum = 0.0;
    double x;
    steps = 1.0 / num_steps;
    omp_set_num_threads(4);
    double start = omp_get_wtime();
    #pragma omp parallel for private(x) reduction(+:sum) schedule(dynamic, 100000)
    for (int i = 0; i < num_steps; i++) {
        x = (i + 0.5) * steps;
        sum += 4.0 / (1.0 + x * x);
    }

    // THE CODE OF CHAOS
    // #pragma omp parallel for shared(sum) 
    // for (int i = 0; i < num_steps; i++) {
    //     x = (i + 0.5) * steps;
    //     // Massive Race Condition: 4 threads trying to += the same address at once
    //     sum += 4.0 / (1.0 + x * x);
    // }

    // int num_threads = 4;
    // double sums[4] = {0.0, 0.0, 0.0, 0.0}; // Array for results
    // #pragma omp parallel
    // {
    //     int id = omp_get_thread_num();
    //     int num_threads = omp_get_num_threads();
    //     for (int i = id; i < num_steps; i += num_threads) {
    //         x = (i + 0.5) * steps;
    //         sums[id] += 4.0 / (1.0 + x * x);
    //     }
    // }
    // for (int i = 0; i < num_threads; i++) {
    //     sum += sums[i];
    // }


    pi = steps * sum;
    double end = omp_get_wtime();
    cout << "Pi = " << pi << endl;
    cout << "Time taken: " << end - start << " seconds" << endl;
    return 0;
}

// #include <iostream>
// #include <omp.h>
// #include <vector>

// #define NUM_STEPS 100000000

// // Struct causing False Sharing
// // All these doubles sit next to each other in memory
// struct BadSum {
//     double vals[4]; // 32 Bytes total. Fits in ONE cache line.
// };
// 
// Struct preventing False Sharing (Padding)
// struct GoodSum {
//     // value (8 bytes) + padding (56 bytes) = 64 Bytes
//     // Each thread gets its OWN cache line!
//     struct {
//         double val;
//         char pad[56]; 
//     } vals[4];
// };

// int main() {
//     int n_threads = 4;
//     omp_set_num_threads(n_threads);
//     double step = 1.0 / NUM_STEPS;
    
//     // TEST 1: FALSE SHARING (The Slow Way)
//     BadSum bad;
//     double start = omp_get_wtime();
//     #pragma omp parallel
//     {
//         int id = omp_get_thread_num();
//         for (int i = id; i < NUM_STEPS; i += n_threads) {
//             double x = (i + 0.5) * step;
//             bad.vals[id] += 4.0 / (1.0 + x * x);
//         }
//     }
//     double end = omp_get_wtime();
//     std::cout << "False Sharing Time: " << (end - start) << " s\n";

//     // TEST 2: PADDING (The Fast Way)
//     GoodSum good;
//     start = omp_get_wtime();
//     #pragma omp parallel
//     {
//         int id = omp_get_thread_num();
//         for (int i = id; i < NUM_STEPS; i += n_threads) {
//             double x = (i + 0.5) * step;
//             good.vals[id].val += 4.0 / (1.0 + x * x);
//         }
//     }
//     end = omp_get_wtime();
//     std::cout << "Padded Time:       " << (end - start) << " s\n";

//     return 0;
// }