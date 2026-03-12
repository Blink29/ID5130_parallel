// q1b.cpp - Monte Carlo Integration (Serial + OpenMP Parallel)
// Integral: I = ∫₀⁴∫₀³∫₀² (4x³ + xy² + 5y + yz + 6z) dz dy dx
//
// Compile serial:   g++ -O2 q1b.cpp -o q1b_serial
// Compile parallel: g++ -O2 -fopenmp q1b.cpp -o q1b_parallel
//
// Run: ./q1b_serial <N>
//      OMP_NUM_THREADS=p ./q1b_parallel <N>
//
// where N = number of random sample points

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Integration limits
const double X_MIN = 0.0, X_MAX = 4.0;
const double Y_MIN = 0.0, Y_MAX = 3.0;
const double Z_MIN = 0.0, Z_MAX = 2.0;

// Volume of the integration domain
const double VOLUME = (X_MAX - X_MIN) * (Y_MAX - Y_MIN) * (Z_MAX - Z_MIN);

// Analytical result of the integral
const double ANALYTICAL = 2040.0;

// The integrand: f(x, y, z) = 4x³ + xy² + 5y + yz + 6z
inline double f(double x, double y, double z) {
    return 4.0 * x * x * x + x * y * y + 5.0 * y + y * z + 6.0 * z;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <N>" << endl;
        cerr << "  N = number of random sample points" << endl;
        return 1;
    }

    long N = atol(argv[1]);

    cout << "=== Monte Carlo Integration ===" << endl;
    cout << "N = " << N << endl;
    cout << "Volume = " << VOLUME << endl;

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    cout << "Running PARALLEL with " << num_threads << " threads" << endl;
#else   
    cout << "Running SERIAL" << endl;
#endif

    double sum_f = 0.0;   // Σ f(x_l, y_l, z_l)
    double sum_f2 = 0.0;  // Σ f(x_l, y_l, z_l)²

    // Start timing
#ifdef _OPENMP
    double t_start = omp_get_wtime();
#else
    clock_t t_start = clock();
#endif

    // Monte Carlo sampling
    // Each thread gets its own random seed based on thread id
#pragma omp parallel reduction(+:sum_f, sum_f2)
    {
        // Seed: use thread-specific seed for reproducibility per thread
        unsigned int seed;
#ifdef _OPENMP
        seed = 42 + omp_get_thread_num();
#else
        seed = 42;
#endif

#pragma omp for
        for (long l = 0; l < N; l++) {
            // Generate random (x, y, z) within the domain
            double x = X_MIN + (X_MAX - X_MIN) * ((double)rand_r(&seed) / RAND_MAX);
            double y = Y_MIN + (Y_MAX - Y_MIN) * ((double)rand_r(&seed) / RAND_MAX);
            double z = Z_MIN + (Z_MAX - Z_MIN) * ((double)rand_r(&seed) / RAND_MAX);

            double val = f(x, y, z);
            sum_f  += val;
            sum_f2 += val * val;
        }
    }

    // End timing
#ifdef _OPENMP
    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;
#else
    clock_t t_end = clock();
    double elapsed = (double)(t_end - t_start) / CLOCKS_PER_SEC;
#endif

    // Compute results using equations (4), (5), (6)
    double f_avg = sum_f / N;               // Eq. (4): f̄
    double f2_avg = sum_f2 / N;             // Eq. (6): f̄²
    double f_rms = sqrt((f2_avg - f_avg * f_avg) / N);  // Eq. (5): f_rms

    double I_mc = VOLUME * f_avg;           // Eq. (3): V * f̄
    double I_err = VOLUME * f_rms;          // Eq. (3): V * f_rms (uncertainty)

    double abs_error = fabs(I_mc - ANALYTICAL);
    double rel_error = (abs_error / ANALYTICAL) * 100.0;

    cout << "\n--- Results ---" << endl;
    cout << "f_avg       = " << f_avg << endl;
    cout << "f_rms       = " << f_rms << endl;
    cout << "I (MC)      = " << I_mc << " ± " << I_err << endl;
    cout << "I (Exact)   = " << ANALYTICAL << endl;
    cout << "Abs Error   = " << abs_error << endl;
    cout << "Rel Error   = " << rel_error << " %" << endl;
    cout << "Time        = " << elapsed << " seconds" << endl;

    return 0;
}
