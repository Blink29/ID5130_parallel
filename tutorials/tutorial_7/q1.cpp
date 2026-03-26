// MPI Finite Difference Derivatives of u(x) = x^3 - sin(5x)
// Computes du/dx using 1st, 2nd and 4th order accurate formulae
//
// Compile: mpic++ -O2 q1.cpp -o q1 -lm
// Run:     mpirun -np <p> ./q1 <dx>
//          e.g. mpirun -np 4 ./q1 0.01

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdlib>

using namespace std;

// u(x) = x^3 - sin(5x)
inline double u(double x) {
    return x * x * x - sin(5.0 * x);
}

// Analytical derivative: du/dx = 3x^2 - 5cos(5x)
inline double dudx_exact(double x) {
    return 3.0 * x * x - 5.0 * cos(5.0 * x);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse dx from command line (default 0.01)
    double dx = 0.01;
    if (argc > 1) {
        dx = atof(argv[1]);
    }

    double x_min = 0.0, x_max = 3.0;
    int N = (int)round((x_max - x_min) / dx); // number of intervals
    int N_pts = N + 1;                          // total grid points

    // Distribute points across processes
    int base_count = N_pts / size;
    int remainder  = N_pts % size;

    // Each process gets base_count points; first 'remainder' get one extra
    vector<int> counts(size), displs(size);
    for (int i = 0; i < size; i++) {
        counts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }

    int local_n  = counts[rank];
    int local_start = displs[rank]; // global index of first local point

    // Allocate local arrays with ghost cells (2 on each side for 4th order)
    int nghost = 2;
    int local_total = local_n + 2 * nghost;
    vector<double> local_u(local_total);
    vector<double> local_x(local_total);

    // Fill local data (indices: nghost ... nghost+local_n-1 are real points)
    for (int i = 0; i < local_n; i++) {
        int gi = local_start + i; // global index
        double xi = x_min + gi * dx;
        local_x[nghost + i] = xi;
        local_u[nghost + i] = u(xi);
    }

    // Exchange ghost points with neighbors
    // We need up to 2 ghost points on each side for the 4th-order stencil
    int left  = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Send/receive 2 points to/from left and right
    // Send leftmost 2 real points to left neighbor's right ghost
    // Receive right ghost from right neighbor
    double send_left[2], send_right[2], recv_left[2], recv_right[2];

    // Prepare data to send
    for (int g = 0; g < nghost; g++) {
        send_left[g]  = local_u[nghost + g];                    // leftmost real points
        send_right[g] = local_u[nghost + local_n - nghost + g]; // rightmost real points
    }

    // Exchange with right neighbor
    MPI_Sendrecv(send_right, nghost, MPI_DOUBLE, right, 0,
                 recv_left,  nghost, MPI_DOUBLE, left,  0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Exchange with left neighbor
    MPI_Sendrecv(send_left,  nghost, MPI_DOUBLE, left,  1,
                 recv_right, nghost, MPI_DOUBLE, right, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Fill ghost cells
    if (left != MPI_PROC_NULL) {
        for (int g = 0; g < nghost; g++)
            local_u[g] = recv_left[g];
    }
    if (right != MPI_PROC_NULL) {
        for (int g = 0; g < nghost; g++)
            local_u[nghost + local_n + g] = recv_right[g];
    }

    // Also fill ghost x-values (for reference, though not strictly needed)
    for (int g = 0; g < nghost; g++) {
        local_x[g] = x_min + (local_start - nghost + g) * dx;
        local_x[nghost + local_n + g] = x_min + (local_start + local_n + g) * dx;
    }

    // Compute derivatives using all four schemes
    vector<double> d_fwd(local_n, 0.0);   // 1st-order forward
    vector<double> d_bwd(local_n, 0.0);   // 1st-order backward
    vector<double> d_c2(local_n, 0.0);    // 2nd-order central
    vector<double> d_c4(local_n, 0.0);    // 4th-order central

    for (int i = 0; i < local_n; i++) {
        int gi = local_start + i; // global index
        int li = nghost + i;      // local index (in ghost-padded array)

        // --- 1st-order forward: (f[i+1] - f[i]) / dx ---
        if (gi < N) {
            d_fwd[i] = (local_u[li + 1] - local_u[li]) / dx;
        } else {
            // At the right boundary, use backward difference
            d_fwd[i] = (local_u[li] - local_u[li - 1]) / dx;
        }

        // --- 1st-order backward: (f[i] - f[i-1]) / dx ---
        if (gi > 0) {
            d_bwd[i] = (local_u[li] - local_u[li - 1]) / dx;
        } else {
            // At the left boundary, use forward difference
            d_bwd[i] = (local_u[li + 1] - local_u[li]) / dx;
        }

        // --- 2nd-order central: (f[i+1] - f[i-1]) / (2*dx) ---
        if (gi > 0 && gi < N) {
            d_c2[i] = (local_u[li + 1] - local_u[li - 1]) / (2.0 * dx);
        } else if (gi == 0) {
            // Left boundary: use forward
            d_c2[i] = (local_u[li + 1] - local_u[li]) / dx;
        } else {
            // Right boundary: use backward
            d_c2[i] = (local_u[li] - local_u[li - 1]) / dx;
        }

        // --- 4th-order central: (-f[i-2] + 8f[i-1] - 8f[i+1] + f[i+2]) ---
        //     i.e. (f[i-2] - 8f[i-1] + 8f[i+1] - f[i+2]) / (12*dx)
        if (gi >= 2 && gi <= N - 2) {
            d_c4[i] = (local_u[li - 2] - 8.0 * local_u[li - 1]
                       + 8.0 * local_u[li + 1] - local_u[li + 2]) / (12.0 * dx);
        } else if (gi > 0 && gi < N) {
            // Near-boundary: fall back to 2nd-order central
            d_c4[i] = (local_u[li + 1] - local_u[li - 1]) / (2.0 * dx);
        } else if (gi == 0) {
            d_c4[i] = (local_u[li + 1] - local_u[li]) / dx;
        } else {
            d_c4[i] = (local_u[li] - local_u[li - 1]) / dx;
        }
    }

    // Gather all results to rank 0
    vector<double> all_x(N_pts), all_fwd(N_pts), all_bwd(N_pts), all_c2(N_pts), all_c4(N_pts);

    // Prepare local x values (just the real points)
    vector<double> local_x_real(local_n);
    for (int i = 0; i < local_n; i++)
        local_x_real[i] = local_x[nghost + i];

    MPI_Gatherv(local_x_real.data(), local_n, MPI_DOUBLE,
                all_x.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(d_fwd.data(), local_n, MPI_DOUBLE,
                all_fwd.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(d_bwd.data(), local_n, MPI_DOUBLE,
                all_bwd.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(d_c2.data(), local_n, MPI_DOUBLE,
                all_c2.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(d_c4.data(), local_n, MPI_DOUBLE,
                all_c4.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0: compute errors and write output
    if (rank == 0) {
        double max_err_fwd = 0, max_err_bwd = 0, max_err_c2 = 0, max_err_c4 = 0;
        // Interior-only errors (exclude boundary points affected by fallback)
        double int_err_fwd = 0, int_err_bwd = 0, int_err_c2 = 0, int_err_c4 = 0;

        for (int i = 0; i < N_pts; i++) {
            double exact = dudx_exact(all_x[i]);
            double e_fwd = fabs(all_fwd[i] - exact);
            double e_bwd = fabs(all_bwd[i] - exact);
            double e_c2  = fabs(all_c2[i]  - exact);
            double e_c4  = fabs(all_c4[i]  - exact);
            max_err_fwd = max(max_err_fwd, e_fwd);
            max_err_bwd = max(max_err_bwd, e_bwd);
            max_err_c2  = max(max_err_c2,  e_c2);
            max_err_c4  = max(max_err_c4,  e_c4);
            if (i >= 2 && i <= N - 2) {
                int_err_fwd = max(int_err_fwd, e_fwd);
                int_err_bwd = max(int_err_bwd, e_bwd);
                int_err_c2  = max(int_err_c2,  e_c2);
                int_err_c4  = max(int_err_c4,  e_c4);
            }
        }

        cout << "=======================================" << endl;
        cout << "  Finite Difference Derivative Results" << endl;
        cout << "  dx = " << dx << ", N = " << N << ", procs = " << size << endl;
        cout << "=======================================" << endl;
        cout << fixed << setprecision(10);
        cout << "  Max Error (overall):" << endl;
        cout << "    1st-order forward:  " << scientific << max_err_fwd << endl;
        cout << "    1st-order backward: " << scientific << max_err_bwd << endl;
        cout << "    2nd-order central:  " << scientific << max_err_c2  << endl;
        cout << "    4th-order central:  " << scientific << max_err_c4  << endl;
        cout << "  Max Error (interior only, i=2..N-2):" << endl;
        cout << "    1st-order forward:  " << scientific << int_err_fwd << endl;
        cout << "    1st-order backward: " << scientific << int_err_bwd << endl;
        cout << "    2nd-order central:  " << scientific << int_err_c2  << endl;
        cout << "    4th-order central:  " << scientific << int_err_c4  << endl;
        cout << "=======================================" << endl;

        // Write CSV file
        string fname = "deriv_dx" + to_string(dx) + ".csv";
        ofstream fout(fname);
        fout << "x,analytical,forward,backward,central2,central4" << endl;
        fout << fixed << setprecision(12);
        for (int i = 0; i < N_pts; i++) {
            fout << all_x[i] << ","
                 << dudx_exact(all_x[i]) << ","
                 << all_fwd[i] << ","
                 << all_bwd[i] << ","
                 << all_c2[i]  << ","
                 << all_c4[i]  << endl;
        }
        fout.close();
        cout << "Results written to " << fname << endl;
    }

    MPI_Finalize();
    return 0;
}
