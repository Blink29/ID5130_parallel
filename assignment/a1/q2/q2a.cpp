#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;

int main() {
    double delta = 0.1;
    int N = (int)(2.0 / delta) + 1; // 21 points along each direction
    double x_min = -1.0, y_min = -1.0;

    // Allocate grids
    vector<vector<double>> phi(N, vector<double>(N, 0.0));
    vector<vector<double>> phi_new(N, vector<double>(N, 0.0));
    vector<vector<double>> q(N, vector<double>(N, 0.0));
    vector<vector<double>> phi_exact(N, vector<double>(N, 0.0));

    // Compute source term q and exact solution
    for (int i = 0; i < N; i++) {
        double x = x_min + i * delta;
        for (int j = 0; j < N; j++) {
            double y = y_min + j * delta;
            q[i][j] = 2.0 * (2.0 - x * x - y * y);
            phi_exact[i][j] = (x * x - 1.0) * (y * y - 1.0);
        }
    }

    // Jacobi iteration
    int max_iter = 1000000;
    int iter = 0;
    bool converged = false;

    for (iter = 1; iter <= max_iter; iter++) {
        // Update interior points
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                phi_new[i][j] = 0.25 * (phi[i + 1][j] + phi[i - 1][j] +
                                         phi[i][j + 1] + phi[i][j - 1]) +
                                (delta * delta / 4.0) * q[i][j];
            }
        }

        // Check convergence: all interior points within 1% of exact
        converged = true;
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                double exact_val = phi_exact[i][j];
                if (fabs(exact_val) > 1e-12) {
                    double rel_error = fabs((phi_new[i][j] - exact_val) / exact_val);
                    if (rel_error > 0.01) {
                        converged = false;
                        break;
                    }
                } else {
                    if (fabs(phi_new[i][j] - exact_val) > 1e-12) {
                        converged = false;
                        break;
                    }
                }
            }
            if (!converged) break;
        }

        // Copy phi_new to phi
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                phi[i][j] = phi_new[i][j];
            }
        }

        if (converged) break;
    }

    cout << "Converged after " << iter << " iterations." << endl;

    // Write data to file
    ofstream outfile("q2a_data.txt");
    outfile << iter << endl;

    // Find j index for y = 0.5
    int j_half = (int)round((0.5 - y_min) / delta);
    double y_val = y_min + j_half * delta;
    cout << "Plotting at y = " << y_val << endl;

    outfile << scientific;
    outfile.precision(12);
    for (int i = 0; i < N; i++) {
        double x = x_min + i * delta;
        outfile << x << " " << phi[i][j_half] << " " << phi_exact[i][j_half] << endl;
    }
    outfile.close();

    cout << "Data written to q2a_data.txt" << endl;
    return 0;
}
