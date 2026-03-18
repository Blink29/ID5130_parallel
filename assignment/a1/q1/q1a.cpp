// Riemann Integration (Serial + OpenMP Parallel)
// Integral: I = ∫₀⁴∫₀³∫₀² (4x³ + xy² + 5y + yz + 6z) dz dy dx
//
// Compile serial:   g++ -O2 q1a.cpp -o q1a_serial
// Compile parallel: g++ -O2 -fopenmp q1a.cpp -o q1a_parallel
//
// Run: ./q1a_serial
//      OMP_NUM_THREADS=p ./q1a_parallel

#include <iostream>
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

// Analytical result of the integral
const double ANALYTICAL = 2040.0;

// The integrand: f(x, y, z) = 4x³ + xy² + 5y + yz + 6z
inline double f(double x, double y, double z) {
    return 4.0 * x * x * x + x * y * y + 5.0 * y + y * z + 6.0 * z;
}

int main() {
    // Number of subdivisions: Nx = 48, Ny = 36, Nz = 24
    int Nx = 48, Ny = 36, Nz = 24;

    double dx = (X_MAX - X_MIN) / Nx;
    double dy = (Y_MAX - Y_MIN) / Ny;
    double dz = (Z_MAX - Z_MIN) / Nz;
    double dv = dx * dy * dz;  // incremental volume Δv

    cout << "=== Riemann Integration ===" << endl;
    cout << "Nx = " << Nx << ", Ny = " << Ny << ", Nz = " << Nz << endl;
    cout << "dx = " << dx << ", dy = " << dy << ", dz = " << dz << endl;
    cout << "dv = " << dv << endl;

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    cout << "Running PARALLEL with " << num_threads << " threads" << endl;
#else
    cout << "Running SERIAL" << endl;
#endif

    double sum = 0.0;

    // Start timing
#ifdef _OPENMP
    double t_start = omp_get_wtime();
#else
    clock_t t_start = clock();
#endif

    // Riemann sum: using midpoint rule for better accuracy
    // x*_ijk = X_MIN + (i + 0.5) * dx, etc.
#pragma omp parallel for reduction(+:sum) collapse(3)
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            for (int k = 0; k < Nz; k++) {
                double x = X_MIN + (i + 0.5) * dx;
                double y = Y_MIN + (j + 0.5) * dy;
                double z = Z_MIN + (k + 0.5) * dz;
                sum += f(x, y, z) * dv;
            }
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

    double abs_error = fabs(sum - ANALYTICAL);
    double rel_error = (abs_error / ANALYTICAL) * 100.0;

    cout << "\n--- Results ---" << endl;
    cout << "I (Riemann) = " << sum << endl;
    cout << "I (Exact)   = " << ANALYTICAL << endl;
    cout << "Abs Error   = " << abs_error << endl;
    cout << "Rel Error   = " << rel_error << " %" << endl;
    cout << "Time        = " << elapsed << " seconds" << endl;

    return 0;
}
