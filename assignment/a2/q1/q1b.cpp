// MPI parallel solver for 1D traveling wave equation
// du/dt + c du/dx = 0, c = 1.0, L = 2.0
// Spatial: (i) First-order upwind, (ii) Third-order QUICK
// Time: Euler explicit
//
// Compile: mpic++ -O2 q1b.cpp -o q1b -lm
// Run:     mpirun -np <p> ./q1b

#include <mpi.h>
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N  = (int)round(L / dx);
    int Np = N + 1;
    double sigma = c_wave * dt / dx;

    if (rank == 0) {
        cout << "=== 1D Wave Equation - MPI (" << size << " procs) ===" << endl;
        cout << "N+1 = " << Np << ", dx = " << dx << ", dt = " << dt << endl;
        cout << "CFL = " << sigma << endl;
    }

    // Distribute grid points across processes
    int base = Np / size;
    int rem  = Np % size;
    vector<int> counts(size), displs(size);
    for (int i = 0; i < size; i++) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }

    int local_n     = counts[rank];
    int local_start = displs[rank]; // global index of first local point

    // Ghost points: 2 on each side (QUICK needs i-2 and i+1)
    int nghost      = 2;
    int local_total = local_n + 2 * nghost;

    // Allocate padded arrays [ghost_L | real | ghost_R]
    vector<double> local_x(local_total);
    vector<double> u_up(local_total, 0.0), u_qk(local_total, 0.0);
    vector<double> u_up_new(local_total, 0.0), u_qk_new(local_total, 0.0);

    // Fill grid coordinates and initial condition
    for (int i = 0; i < local_total; i++)
        local_x[i] = (local_start - nghost + i) * dx;
    for (int i = 0; i < local_n; i++) {
        int li = nghost + i;
        u_up[li] = u0(local_x[li]);
        u_qk[li] = u0(local_x[li]);
    }

    int left  = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    double t_out[] = {0.0, 0.5, 1.0};
    int n_out = 3;

    vector<vector<double>> exact_out(n_out, vector<double>(Np));
    vector<vector<double>> upwind_out(n_out, vector<double>(Np));
    vector<vector<double>> quick_out(n_out, vector<double>(Np));

    // Helper: gather real data to rank 0
    auto gather = [&](int k, double t_curr) {
        vector<double> loc_up(local_n), loc_qk(local_n);
        for (int i = 0; i < local_n; i++) {
            loc_up[i] = u_up[nghost + i];
            loc_qk[i] = u_qk[nghost + i];
        }
        MPI_Gatherv(loc_up.data(), local_n, MPI_DOUBLE,
                     upwind_out[k].data(), counts.data(), displs.data(),
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(loc_qk.data(), local_n, MPI_DOUBLE,
                     quick_out[k].data(), counts.data(), displs.data(),
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0)
            for (int i = 0; i < Np; i++)
                exact_out[k][i] = u_exact(i * dx, t_curr);
    };

    gather(0, 0.0); // t = 0
    int out_idx = 1;

    int total_steps = (int)round(t_out[n_out - 1] / dt);

    for (int n = 0; n < total_steps; n++) {
        double t_curr = (n + 1) * dt;

        // --- Ghost exchange for u_up ---
        double sL[2], sR[2], rL[2], rR[2];
        for (int g = 0; g < nghost; g++) {
            sL[g] = u_up[nghost + g];
            sR[g] = u_up[nghost + local_n - nghost + g];
        }
        MPI_Sendrecv(sR, nghost, MPI_DOUBLE, right, 0,
                     rL, nghost, MPI_DOUBLE, left,  0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(sL, nghost, MPI_DOUBLE, left,  1,
                     rR, nghost, MPI_DOUBLE, right, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (left  != MPI_PROC_NULL)
            for (int g = 0; g < nghost; g++) u_up[g] = rL[g];
        if (right != MPI_PROC_NULL)
            for (int g = 0; g < nghost; g++) u_up[nghost + local_n + g] = rR[g];

        // --- Ghost exchange for u_qk ---
        for (int g = 0; g < nghost; g++) {
            sL[g] = u_qk[nghost + g];
            sR[g] = u_qk[nghost + local_n - nghost + g];
        }
        MPI_Sendrecv(sR, nghost, MPI_DOUBLE, right, 2,
                     rL, nghost, MPI_DOUBLE, left,  2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(sL, nghost, MPI_DOUBLE, left,  3,
                     rR, nghost, MPI_DOUBLE, right, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (left  != MPI_PROC_NULL)
            for (int g = 0; g < nghost; g++) u_qk[g] = rL[g];
        if (right != MPI_PROC_NULL)
            for (int g = 0; g < nghost; g++) u_qk[nghost + local_n + g] = rR[g];

        // --- Update local points ---
        for (int i = 0; i < local_n; i++) {
            int gi = local_start + i;
            int li = nghost + i;

            if (gi == 0 || gi == N) {
                u_up_new[li] = 0.0;
                u_qk_new[li] = 0.0;
                continue;
            }

            // Upwind
            u_up_new[li] = u_up[li] - sigma * (u_up[li] - u_up[li-1]);

            // QUICK (upwind at gi==1 where i-2 is unavailable)
            if (gi == 1) {
                u_qk_new[li] = u_qk[li] - sigma * (u_qk[li] - u_qk[li-1]);
            } else {
                double dudx = (3.0/8.0)*u_qk[li] - (7.0/8.0)*u_qk[li-1]
                            + (1.0/8.0)*u_qk[li-2] + (3.0/8.0)*u_qk[li+1];
                u_qk_new[li] = u_qk[li] - sigma * dudx;
            }
        }

        for (int i = 0; i < local_n; i++) {
            int li = nghost + i;
            u_up[li] = u_up_new[li];
            u_qk[li] = u_qk_new[li];
        }

        if (out_idx < n_out) {
            int target = (int)round(t_out[out_idx] / dt);
            if (n + 1 == target) {
                gather(out_idx, t_curr);
                if (rank == 0) cout << "Saved at t = " << t_curr << endl;
                out_idx++;
            }
        }
    }

    if (rank == 0) {
        string fname = "q1b_p" + to_string(size) + "_data.csv";
        ofstream fout(fname);
        fout << "x";
        for (int k = 0; k < n_out; k++)
            fout << ",exact_t" << t_out[k] << ",upwind_t" << t_out[k] << ",quick_t" << t_out[k];
        fout << endl;
        fout << fixed << setprecision(10);
        for (int i = 0; i < Np; i++) {
            fout << i * dx;
            for (int k = 0; k < n_out; k++)
                fout << "," << exact_out[k][i] << "," << upwind_out[k][i] << "," << quick_out[k][i];
            fout << endl;
        }
        fout.close();
        cout << "Results written to " << fname << endl;
    }

    MPI_Finalize();
    return 0;
}
