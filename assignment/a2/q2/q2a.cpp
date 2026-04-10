// Serial Jacobi solver for 2D Poisson equation
// ∇²φ = -(x²+y²) on [-1,1]×[-1,1]
// BCs: φ=0 (top,bottom), φ=sin(2πy) (left), ∂φ/∂x=0 (right)
//
// Compile: g++ -O2 q2a.cpp -o q2a -lm
// Run:     ./q2a [delta]   (default delta=0.1)

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

using namespace std;
const double PI = M_PI;

int main(int argc, char* argv[]) {
    double delta = 0.1;
    if (argc > 1) delta = atof(argv[1]);

    int N  = (int)round(2.0 / delta);
    int Np = N + 1;                    // grid points per direction
    double d2 = delta * delta;

    cout << "=== Poisson - Serial Jacobi ===" << endl;
    cout << "delta=" << delta << "  grid=" << Np << "x" << Np << endl;

    vector<double> x(Np), y(Np);
    for (int i = 0; i < Np; i++) x[i] = -1.0 + i * delta;
    for (int j = 0; j < Np; j++) y[j] = -1.0 + j * delta;

    // phi stored row-major: phi[j*Np + i]
    int tot = Np * Np;
    vector<double> phi(tot, 0.0), pn(tot, 0.0);

    // Left BC: φ(x=-1) = sin(2πy)
    for (int j = 0; j < Np; j++) {
        double v = sin(2.0 * PI * y[j]);
        phi[j * Np] = v;
        pn[j * Np]  = v;
    }

    double tol = 1e-4;
    int iter;
    double err;
    clock_t t0 = clock();

    for (iter = 1; iter <= 2000000; iter++) {
        double err_sq = 0.0;

        // Jacobi update for interior points
        for (int j = 1; j < N; j++)
            for (int i = 1; i < N; i++) {
                double v = 0.25 * (phi[j*Np+i+1] + phi[j*Np+i-1]
                                 + phi[(j+1)*Np+i] + phi[(j-1)*Np+i])
                         + (d2 / 4.0) * (x[i]*x[i] + y[j]*y[j]);
                pn[j*Np+i] = v;
            }

        // Neumann BC at x=1: φ_N = (4φ_{N-1} - φ_{N-2})/3
        for (int j = 1; j < N; j++)
            pn[j*Np+N] = (4.0*pn[j*Np+N-1] - pn[j*Np+N-2]) / 3.0;

        // Convergence: L2 norm of successive-iteration difference
        for (int k = 0; k < tot; k++) {
            double d = fabs(pn[k] - phi[k]);
            err_sq += d * d;
        }
        err = sqrt(err_sq);

        swap(phi, pn);
        if (err < tol) break;
        if (iter % 10000 == 0) cout << "  iter " << iter << "  err=" << err << endl;
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    cout << "Converged: " << iter << " iterations, err=" << err
         << ", time=" << elapsed << "s" << endl;

    // Output φ vs x at y=0, φ vs y at x=0
    int jy0 = N / 2, ix0 = N / 2;
    string tag = to_string(delta);
    ofstream fout("q2a_d" + tag + "_data.csv");
    fout << "coord,phi_vs_x_at_y0,phi_vs_y_at_x0" << endl;
    fout << fixed << setprecision(10);
    for (int k = 0; k < Np; k++)
        fout << (-1.0 + k*delta) << "," << phi[jy0*Np+k] << "," << phi[k*Np+ix0] << endl;
    fout.close();
    cout << "Written q2a_d" + tag + "_data.csv" << endl;

    return 0;
}
