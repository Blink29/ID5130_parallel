// MPI Jacobi solver for 2D Poisson equation
// ∇²φ = -(x²+y²) on [-1,1]×[-1,1]
// BCs: φ=0 (top,bottom), φ=sin(2πy) (left), ∂φ/∂x=0 (right)
// Row-wise (y-direction) 1D domain decomposition
//
// Compile: mpic++ -O2 q2b.cpp -o q2b -lm
// Run:     mpirun -np <p> ./q2b [delta]   (default delta=0.01)

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdlib>

using namespace std;
const double PI = M_PI;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    double delta = 0.01;
    if (argc > 1) delta = atof(argv[1]);

    int N  = (int)round(2.0 / delta);
    int Np = N + 1;
    double d2 = delta * delta;

    if (rank == 0) {
        cout << "=== Poisson - MPI Jacobi (" << nproc << " procs) ===" << endl;
        cout << "delta=" << delta << "  grid=" << Np << "x" << Np << endl;
    }

    vector<double> x(Np), y(Np);
    for (int i = 0; i < Np; i++) x[i] = -1.0 + i * delta;
    for (int j = 0; j < Np; j++) y[j] = -1.0 + j * delta;

    // Distribute rows (y-direction)
    int base = Np / nproc, rem = Np % nproc;
    vector<int> cnts(nproc), disp(nproc);
    for (int r = 0; r < nproc; r++) {
        cnts[r] = base + (r < rem ? 1 : 0);
        disp[r] = (r == 0) ? 0 : disp[r-1] + cnts[r-1];
    }
    int lny = cnts[rank];       // local row count
    int j0  = disp[rank];       // global j of first owned row

    // Local array: (lny+2) rows × Np cols, 1 ghost row each side
    // jl=0: ghost bottom, jl=1..lny: real, jl=lny+1: ghost top
    int ltot = (lny + 2) * Np;
    vector<double> phi(ltot, 0.0), pn(ltot, 0.0);

    // Left BC: φ(x=-1) = sin(2πy)
    for (int jl = 1; jl <= lny; jl++) {
        double v = sin(2.0 * PI * y[j0 + jl - 1]);
        phi[jl * Np] = v;
        pn[jl * Np]  = v;
    }

    int below = (rank > 0)         ? rank - 1 : MPI_PROC_NULL;
    int above = (rank < nproc - 1) ? rank + 1 : MPI_PROC_NULL;

    double tol = 1e-4, err, gerr;
    int iter;
    double t0 = MPI_Wtime();

    for (iter = 1; iter <= 2000000; iter++) {
        // Ghost exchange: send top real→above, recv from below→ghost bottom
        MPI_Sendrecv(&phi[lny * Np], Np, MPI_DOUBLE, above, 0,
                     &phi[0],        Np, MPI_DOUBLE, below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // Send bottom real→below, recv from above→ghost top
        MPI_Sendrecv(&phi[1 * Np],       Np, MPI_DOUBLE, below, 1,
                     &phi[(lny+1)*Np],   Np, MPI_DOUBLE, above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double err_sq = 0.0;
        for (int jl = 1; jl <= lny; jl++) {
            int gj = j0 + jl - 1;
            if (gj == 0 || gj == N) continue;  // Dirichlet rows

            for (int i = 1; i < N; i++) {
                double v = 0.25 * (phi[jl*Np+i+1] + phi[jl*Np+i-1]
                                 + phi[(jl+1)*Np+i] + phi[(jl-1)*Np+i])
                         + (d2 / 4.0) * (x[i]*x[i] + y[gj]*y[gj]);
                pn[jl*Np+i] = v;
            }
            // Neumann at x=1
            pn[jl*Np+N] = (4.0*pn[jl*Np+N-1] - pn[jl*Np+N-2]) / 3.0;
        }

        // Convergence: L2 norm of successive-iteration difference
        for (int jl = 1; jl <= lny; jl++)
            for (int i = 0; i < Np; i++) {
                double d = pn[jl*Np+i] - phi[jl*Np+i];
                err_sq += d * d;
            }
        err = err_sq;

        MPI_Allreduce(&err, &gerr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        gerr = sqrt(gerr);

        // Copy new→old (real rows only)
        for (int jl = 1; jl <= lny; jl++)
            for (int i = 0; i < Np; i++)
                phi[jl*Np+i] = pn[jl*Np+i];

        if (gerr < tol) break;
        if (rank == 0 && iter % 10000 == 0)
            cout << "  iter " << iter << "  err=" << gerr << endl;
    }

    double elapsed = MPI_Wtime() - t0;
    if (rank == 0)
        cout << "Converged: " << iter << " iters, err=" << gerr
             << ", time=" << elapsed << "s" << endl;

    // --- Gather output data ---
    // φ vs x at y=0 (row j=N/2)
    int jy0 = N / 2;
    vector<double> row_y0(Np, 0.0);
    if (jy0 >= j0 && jy0 < j0 + lny) {
        int jl = jy0 - j0 + 1;
        for (int i = 0; i < Np; i++) row_y0[i] = phi[jl*Np+i];
    }
    vector<double> grow_y0(Np, 0.0);
    MPI_Reduce(row_y0.data(), grow_y0.data(), Np, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);

    // φ vs y at x=0 (column i=N/2)
    int ix0 = N / 2;
    vector<double> lcol(lny);
    for (int jl = 1; jl <= lny; jl++) lcol[jl-1] = phi[jl*Np+ix0];

    vector<double> gcol(Np, 0.0);
    MPI_Gatherv(lcol.data(), lny, MPI_DOUBLE,
                gcol.data(), cnts.data(), disp.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        string fn = "q2b_p" + to_string(nproc) + "_data.csv";
        ofstream fout(fn);
        fout << "coord,phi_vs_x_at_y0,phi_vs_y_at_x0" << endl;
        fout << fixed << setprecision(10);
        for (int k = 0; k < Np; k++)
            fout << (-1.0 + k*delta) << "," << grow_y0[k] << "," << gcol[k] << endl;
        fout.close();
        cout << "Written " << fn << endl;
    }

    MPI_Finalize();
    return 0;
}
