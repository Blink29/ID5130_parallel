#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <omp.h>

// u(x) = 7 - x * tan(x)
double u(double x) {
    return 7.0 - x * tan(x);
}

// Analytical derivative: u'(x) = -tan(x) - x * sec^2(x)
double u_prime_exact(double x) {
    double sec_x = 1.0 / cos(x);
    return -tan(x) - x * sec_x * sec_x;
}

int main() {
    double dx_values[] = {0.01, 0.001};
    int num_dx = 2;
    int thread_counts[] = {2, 4, 8};
    int num_threads_tests = 3;

    printf("==========================================================\n");
    printf("  Finite Difference Derivative of u(x) = 7 - x*tan(x)\n");
    printf("  Analytical: u'(x) = -tan(x) - x*sec^2(x)\n");
    printf("  Domain: x in [-1, 1]\n");
    printf("==========================================================\n");

    for (int d = 0; d < num_dx; d++) {
        double dx = dx_values[d];
        int N = (int)round(2.0 / dx) + 1;  // number of grid points from -1 to 1

        // Allocate arrays
        double* x     = (double*)malloc(N * sizeof(double));
        double* exact  = (double*)malloc(N * sizeof(double));
        double* fwd    = (double*)malloc(N * sizeof(double));  // 1st-order forward  (eq 2)
        double* bwd    = (double*)malloc(N * sizeof(double));  // 1st-order backward (eq 3)
        double* cen2   = (double*)malloc(N * sizeof(double));  // 2nd-order central  (eq 4)
        double* cen4   = (double*)malloc(N * sizeof(double));  // 4th-order central  (eq 5)

        // Fill grid and exact solution (serial)
        for (int i = 0; i < N; i++) {
            x[i] = -1.0 + i * dx;
            exact[i] = u_prime_exact(x[i]);
        }

        for (int t = 0; t < num_threads_tests; t++) {
            int nthreads = thread_counts[t];
            omp_set_num_threads(nthreads);

            printf("\n----------------------------------------------------------\n");
            printf("  dx = %.4f  |  Grid points N = %d  |  Threads = %d\n", dx, N, nthreads);
            printf("----------------------------------------------------------\n");

            // ---- 1st-order Forward (Eq. 2) ----
            double t_start = omp_get_wtime();
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                if (i < N - 1)
                    fwd[i] = (u(x[i] + dx) - u(x[i])) / dx;
                else  // right boundary: use backward
                    fwd[i] = (u(x[i]) - u(x[i] - dx)) / dx;
            }
            double t_fwd = omp_get_wtime() - t_start;

            // ---- 1st-order Backward (Eq. 3) ----
            t_start = omp_get_wtime();
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                if (i > 0)
                    bwd[i] = (u(x[i]) - u(x[i] - dx)) / dx;
                else  // left boundary: use forward
                    bwd[i] = (u(x[i] + dx) - u(x[i])) / dx;
            }
            double t_bwd = omp_get_wtime() - t_start;

            // ---- 2nd-order Central (Eq. 4) ----
            // Boundaries: use 1st-order forward/backward
            t_start = omp_get_wtime();
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                if (i == 0)
                    cen2[i] = (u(x[i] + dx) - u(x[i])) / dx;
                else if (i == N - 1)
                    cen2[i] = (u(x[i]) - u(x[i] - dx)) / dx;
                else
                    cen2[i] = (u(x[i] + dx) - u(x[i] - dx)) / (2.0 * dx);
            }
            double t_cen2 = omp_get_wtime() - t_start;

            // ---- 4th-order Central (Eq. 5) ----
            // Boundary & near-boundary (i=0,1,N-2,N-1): use 1st-order formulas
            t_start = omp_get_wtime();
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                if (i == 0)
                    cen4[i] = (u(x[i] + dx) - u(x[i])) / dx;
                else if (i == N - 1)
                    cen4[i] = (u(x[i]) - u(x[i] - dx)) / dx;
                else if (i == 1 || i == N - 2)
                    cen4[i] = (u(x[i] + dx) - u(x[i] - dx)) / (2.0 * dx);
                else
                    cen4[i] = (u(x[i] - 2*dx) - 8.0*u(x[i] - dx) + 8.0*u(x[i] + dx) - u(x[i] + 2*dx)) / (12.0 * dx);
            }
            double t_cen4 = omp_get_wtime() - t_start;

            // ---- Compute max absolute errors ----
            double err_fwd = 0, err_bwd = 0, err_cen2 = 0, err_cen4 = 0;
            for (int i = 0; i < N; i++) {
                double ef = fabs(fwd[i] - exact[i]);
                double eb = fabs(bwd[i] - exact[i]);
                double e2 = fabs(cen2[i] - exact[i]);
                double e4 = fabs(cen4[i] - exact[i]);
                if (ef > err_fwd) err_fwd = ef;
                if (eb > err_bwd) err_bwd = eb;
                if (e2 > err_cen2) err_cen2 = e2;
                if (e4 > err_cen4) err_cen4 = e4;
            }

            printf("  %-28s Time: %10.6f s  |  Max Error: %e\n", "1st-order Forward (Eq.2):", t_fwd, err_fwd);
            printf("  %-28s Time: %10.6f s  |  Max Error: %e\n", "1st-order Backward (Eq.3):", t_bwd, err_bwd);
            printf("  %-28s Time: %10.6f s  |  Max Error: %e\n", "2nd-order Central (Eq.4):", t_cen2, err_cen2);
            printf("  %-28s Time: %10.6f s  |  Max Error: %e\n", "4th-order Central (Eq.5):", t_cen4, err_cen4);

            // Print a sample of values at a few points
            printf("\n  Sample comparison at selected points:\n");
            printf("  %10s  %14s  %14s  %14s  %14s  %14s\n",
                   "x", "Exact", "Fwd(O1)", "Bwd(O1)", "Cen(O2)", "Cen(O4)");
            int step = (N - 1) / 10;
            if (step < 1) step = 1;
            for (int i = 0; i < N; i += step) {
                printf("  %10.4f  %14.8f  %14.8f  %14.8f  %14.8f  %14.8f\n",
                       x[i], exact[i], fwd[i], bwd[i], cen2[i], cen4[i]);
            }
            // Print last point
            if ((N - 1) % step != 0) {
                int i = N - 1;
                printf("  %10.4f  %14.8f  %14.8f  %14.8f  %14.8f  %14.8f\n",
                       x[i], exact[i], fwd[i], bwd[i], cen2[i], cen4[i]);
            }
        }

        free(x); free(exact);
        free(fwd); free(bwd);
        free(cen2); free(cen4);
    }

    return 0;
}
