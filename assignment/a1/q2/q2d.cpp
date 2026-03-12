// q2d.cpp - Jacobi Solver Thread Scaling Study (Δ = 0.005)
// Part (d): Time vs Number of Threads (p = 1(serial), 2, 4, 8, 16)
//
// Compile: g++ -O2 -fopenmp q2d.cpp -o q2d
// Run:     ./q2d

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <ctime>
#include <omp.h>

using namespace std;

// Serial Jacobi solver (no OpenMP at all)
int jacobi_serial(double delta, double &elapsed_time) {
    int N = (int)(2.0 / delta) + 1;
    double x_min = -1.0, y_min = -1.0;
    double delta2_over4 = delta * delta / 4.0;

    vector<vector<double>> phi(N, vector<double>(N, 0.0));
    vector<vector<double>> phi_new(N, vector<double>(N, 0.0));
    vector<vector<double>> q(N, vector<double>(N, 0.0));
    vector<vector<double>> phi_exact(N, vector<double>(N, 0.0));

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

    clock_t t_start = clock();

    for (iter = 1; iter <= max_iter; iter++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                phi_new[i][j] = 0.25 * (phi[i + 1][j] + phi[i - 1][j] +
                                         phi[i][j + 1] + phi[i][j - 1]) +
                                delta2_over4 * q[i][j];
            }
        }

        converged_flag = 1;
        for (int i = 1; i < N - 1 && converged_flag; i++) {
            for (int j = 1; j < N - 1 && converged_flag; j++) {
                double exact_val = phi_exact[i][j];
                if (fabs(exact_val) > 1e-12) {
                    if (fabs((phi_new[i][j] - exact_val) / exact_val) > 0.01)
                        converged_flag = 0;
                } else {
                    if (fabs(phi_new[i][j] - exact_val) > 1e-12)
                        converged_flag = 0;
                }
            }
        }

        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                phi[i][j] = phi_new[i][j];
            }
        }

        if (converged_flag) break;
    }

    clock_t t_end = clock();
    elapsed_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    return iter;
}

// Parallel Jacobi solver (OpenMP with specified threads)
int jacobi_parallel(double delta, int num_threads, double &elapsed_time) {
    int N = (int)(2.0 / delta) + 1;
    double x_min = -1.0, y_min = -1.0;
    double delta2_over4 = delta * delta / 4.0;

    vector<vector<double>> phi(N, vector<double>(N, 0.0));
    vector<vector<double>> phi_new(N, vector<double>(N, 0.0));
    vector<vector<double>> q(N, vector<double>(N, 0.0));
    vector<vector<double>> phi_exact(N, vector<double>(N, 0.0));

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

    double t_start = omp_get_wtime();

    for (iter = 1; iter <= max_iter; iter++) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                phi_new[i][j] = 0.25 * (phi[i + 1][j] + phi[i - 1][j] +
                                         phi[i][j + 1] + phi[i][j - 1]) +
                                delta2_over4 * q[i][j];
            }
        }

        converged_flag = 1;
        for (int i = 1; i < N - 1 && converged_flag; i++) {
            for (int j = 1; j < N - 1 && converged_flag; j++) {
                double exact_val = phi_exact[i][j];
                if (fabs(exact_val) > 1e-12) {
                    if (fabs((phi_new[i][j] - exact_val) / exact_val) > 0.01)
                        converged_flag = 0;
                } else {
                    if (fabs(phi_new[i][j] - exact_val) > 1e-12)
                        converged_flag = 0;
                }
            }
        }

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                phi[i][j] = phi_new[i][j];
            }
        }

        if (converged_flag) break;
    }

    double t_end = omp_get_wtime();
    elapsed_time = t_end - t_start;
    return iter;
}

int main() {
    double delta = 0.005;
    int threads[] = {2, 4, 8, 16};
    int num_configs = 4;

    ofstream outfile("q2d_data.txt");
    outfile << scientific;
    outfile.precision(12);

    // Serial run
    double t_serial;
    int iter_serial = jacobi_serial(delta, t_serial);
    cout << "Serial: " << iter_serial << " iterations, " << t_serial << " s" << endl;
    outfile << 1 << " " << t_serial << endl;

    // Parallel runs
    for (int t = 0; t < num_configs; t++) {
        double t_par;
        int iter_par = jacobi_parallel(delta, threads[t], t_par);
        cout << "Threads=" << threads[t] << ": " << iter_par
             << " iterations, " << t_par << " s" << endl;
        outfile << threads[t] << " " << t_par << endl;
    }

    outfile.close();
    cout << "Data written to q2d_data.txt" << endl;
    return 0;
}
