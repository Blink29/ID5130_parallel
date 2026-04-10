// Serial solver for 1D traveling wave equation
// du/dt + c du/dx = 0, c = 1.0, L = 2.0
// Spatial: (i) First-order upwind, (ii) Third-order QUICK
// Time: Euler explicit
//
// Compile: g++ -O2 q1a.cpp -o q1a -lm
// Run:     ./q1a

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace std;

const double c_wave = 1.0;
const double L      = 2.0;
const double dx     = 0.002;
const double dt     = 0.0001;
const double PI     = M_PI;

inline double u0(double x) {
    if (x >= 0.0 && x <= 0.5)
        return sin(4.0 * PI * x);
    return 0.0;
}

inline double u_exact(double x, double t) {
    return u0(x - c_wave * t);
}

int main() {
    int N  = (int)round(L / dx);
    int Np = N + 1;
    double sigma = c_wave * dt / dx;

    cout << "=== 1D Wave Equation - Serial ===" << endl;
    cout << "N+1 = " << Np << ", dx = " << dx << ", dt = " << dt << endl;
    cout << "CFL = " << sigma << endl;

    vector<double> x(Np);
    for (int i = 0; i < Np; i++) x[i] = i * dx;

    vector<double> u_up(Np, 0.0), u_qk(Np, 0.0);
    vector<double> u_up_new(Np, 0.0), u_qk_new(Np, 0.0);

    for (int i = 0; i < Np; i++) {
        u_up[i] = u0(x[i]);
        u_qk[i] = u0(x[i]);
    }

    double t_out[] = {0.0, 0.5, 1.0};
    int n_out = 3;

    vector<vector<double>> exact_out(n_out, vector<double>(Np));
    vector<vector<double>> upwind_out(n_out, vector<double>(Np));
    vector<vector<double>> quick_out(n_out, vector<double>(Np));

    for (int i = 0; i < Np; i++) {
        exact_out[0][i]  = u_exact(x[i], 0.0);
        upwind_out[0][i] = u_up[i];
        quick_out[0][i]  = u_qk[i];
    }
    int out_idx = 1;

    int total_steps = (int)round(t_out[n_out - 1] / dt);

    for (int n = 0; n < total_steps; n++) {
        double t_curr = (n + 1) * dt;

        // Upwind
        u_up_new[0] = 0.0;
        u_up_new[N] = 0.0;
        for (int i = 1; i < N; i++)
            u_up_new[i] = u_up[i] - sigma * (u_up[i] - u_up[i - 1]);

        // QUICK
        u_qk_new[0] = 0.0;
        u_qk_new[N] = 0.0;
        u_qk_new[1] = u_qk[1] - sigma * (u_qk[1] - u_qk[0]); // upwind at i=1
        for (int i = 2; i < N; i++) {
            double dudx = (3.0/8.0)*u_qk[i] - (7.0/8.0)*u_qk[i-1]
                        + (1.0/8.0)*u_qk[i-2] + (3.0/8.0)*u_qk[i+1];
            u_qk_new[i] = u_qk[i] - sigma * dudx;
        }

        u_up = u_up_new;
        u_qk = u_qk_new;

        if (out_idx < n_out) {
            int target = (int)round(t_out[out_idx] / dt);
            if (n + 1 == target) {
                for (int i = 0; i < Np; i++) {
                    exact_out[out_idx][i]  = u_exact(x[i], t_curr);
                    upwind_out[out_idx][i] = u_up[i];
                    quick_out[out_idx][i]  = u_qk[i];
                }
                cout << "Saved at t = " << t_curr << endl;
                out_idx++;
            }
        }
    }

    ofstream fout("q1a_data.csv");
    fout << "x";
    for (int k = 0; k < n_out; k++)
        fout << ",exact_t" << t_out[k] << ",upwind_t" << t_out[k] << ",quick_t" << t_out[k];
    fout << endl;
    fout << fixed << setprecision(10);
    for (int i = 0; i < Np; i++) {
        fout << x[i];
        for (int k = 0; k < n_out; k++)
            fout << "," << exact_out[k][i] << "," << upwind_out[k][i] << "," << quick_out[k][i];
        fout << endl;
    }
    fout.close();
    cout << "Results written to q1a_data.csv" << endl;

    return 0;
}
