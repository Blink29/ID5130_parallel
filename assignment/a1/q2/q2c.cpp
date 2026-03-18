// Jacobi Solver for Poisson Equation (Serial + OpenMP Parallel)
// Part (b): OpenMP Jacobi program
// Part (c): Verification + Timing for Δ = 0.1, 0.01, 0.005
//
// Compile serial:   g++ -O2 q2c.cpp -o q2c_serial
// Compile parallel: g++ -O2 -fopenmp q2c.cpp -o q2c_parallel
//
// Run serial:  ./q2c_serial
// Run parallel: OMP_NUM_THREADS=8 ./q2c_parallel

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Jacobi solver - returns iterations, writes elapsed time
// Uses OpenMP if compiled with -fopenmp, otherwise purely serial
int jacobi_solve(double delta, double &elapsed_time,
                 vector<vector<double>> &phi_out) {
    int N = (int)(2.0 / delta) + 1;
    double x_min = -1.0, y_min = -1.0;
    double delta2_over4 = delta * delta / 4.0;

    vector<vector<double>> phi(N, vector<double>(N, 0.0));
    vector<vector<double>> phi_new(N, vector<double>(N, 0.0));
    vector<vector<double>> q(N, vector<double>(N, 0.0));
    vector<vector<double>> phi_exact(N, vector<double>(N, 0.0));

    // Compute source and exact solution
    for (int i = 0; i < N; i++) {
        double x = x_min + i * delta;
        for (int j = 0; j < N; j++) {
            double y = y_min + j * delta;
            q[i][j] = 2.0 * (2.0 - x * x - y * y);
            phi_exact[i][j] = (x * x - 1.0) * (y * y - 1.0);
        }
    }

    int max_iter = 1000000;
    int iter = 0;
    int converged_flag = 0;

    // Start timing: chrono for serial build, omp_get_wtime for OpenMP build
#ifdef _OPENMP
    double t_start = omp_get_wtime();
#else
    auto t_start = chrono::steady_clock::now();
#endif

    for (iter = 1; iter <= max_iter; iter++) {
        // Update interior points
#pragma omp parallel for schedule(static)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                phi_new[i][j] = 0.25 * (phi[i + 1][j] + phi[i - 1][j] +
                                         phi[i][j + 1] + phi[i][j - 1]) +
                                delta2_over4 * q[i][j];
            }
        }

        // Check convergence: all interior points within 1% of exact
        converged_flag = 1;
        for (int i = 1; i < N - 1 && converged_flag; i++) {
            for (int j = 1; j < N - 1 && converged_flag; j++) {
                double exact_val = phi_exact[i][j];
                if (fabs(exact_val) > 1e-12) {
                    double rel_error = fabs((phi_new[i][j] - exact_val) / exact_val);
                    if (rel_error > 0.01) {
                        converged_flag = 0;
                    }
                } else {
                    if (fabs(phi_new[i][j] - exact_val) > 1e-12) {
                        converged_flag = 0;
                    }
                }
            }
        }

        // Copy phi_new -> phi
#pragma omp parallel for schedule(static)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                phi[i][j] = phi_new[i][j];
            }
        }

        if (converged_flag) break;
    }

    // End timing: chrono for serial build, omp_get_wtime for OpenMP build
#ifdef _OPENMP
    double t_end = omp_get_wtime();
    elapsed_time = t_end - t_start;
#else
    auto t_end = chrono::steady_clock::now();
    elapsed_time = chrono::duration<double>(t_end - t_start).count();
#endif

    phi_out = phi;
    return iter;
}

int main() {
#ifdef _OPENMP
    ofstream outfile("q2c_parallel_data.txt");
#else
    ofstream outfile("q2c_serial_data.txt");
#endif
    outfile << scientific;
    outfile.precision(12);

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    cout << "Running PARALLEL with " << num_threads << " threads" << endl;
#else
    cout << "Running SERIAL" << endl;
#endif

    double deltas[] = {0.1, 0.01, 0.005};
    int num_deltas = 3;

    // For verification at Δ=0.1: output φ values at y=0.5
    // (only meaningful when comparing serial and parallel output files)
    double delta_verify = 0.1;
    int N_verify = (int)(2.0 / delta_verify) + 1;
    double x_min = -1.0, y_min = -1.0;
    int j_half = (int)round((0.5 - y_min) / delta_verify);

    double t_elapsed;
    vector<vector<double>> phi;
    int iters = jacobi_solve(delta_verify, t_elapsed, phi);

    // Write verification data
    outfile << "VERIFICATION" << endl;
    outfile << N_verify << endl;
    for (int i = 0; i < N_verify; i++) {
        double x = x_min + i * delta_verify;
        outfile << x << " " << phi[i][j_half] << endl;
    }

    cout << "Delta=0.1: " << iters << " iterations, time=" << t_elapsed << " s" << endl;

    // Timing for all deltas
    outfile << "TIMING" << endl;
    outfile << num_deltas << endl;

    // Reuse delta=0.1 result
    outfile << delta_verify << " " << N_verify << " " << t_elapsed << endl;

    // Run for remaining deltas
    for (int d = 1; d < num_deltas; d++) {
        double delta = deltas[d];
        int N = (int)(2.0 / delta) + 1;

        vector<vector<double>> phi_tmp;
        double t;
        int it = jacobi_solve(delta, t, phi_tmp);

        cout << "Delta=" << delta << " (N=" << N << "): "
             << it << " iterations, time=" << t << " s" << endl;

        outfile << delta << " " << N << " " << t << endl;
    }

    outfile.close();
#ifdef _OPENMP
    cout << "Data written to q2c_parallel_data.txt" << endl;
#else
    cout << "Data written to q2c_serial_data.txt" << endl;
#endif
    return 0;
}
