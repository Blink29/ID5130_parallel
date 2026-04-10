#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 512
#define TOL 1e-6

void matmul_serial(double A[N][N], double B[N][N], double C[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

void matmul_acc(double A[N][N], double B[N][N], double C[N][N]) {
    #pragma acc data copyin(A[0:N][0:N], B[0:N][0:N]) copyout(C[0:N][0:N])
    {
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                #pragma acc loop reduction(+:sum)
                for (int k = 0; k < N; k++)
                    sum += A[i][k] * B[k][j];
                C[i][j] = sum;
            }
    }
}

int main() {
    static double A[N][N], B[N][N];
    static double C_serial[N][N], C_acc[N][N];

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(rand() % 100) / 10.0;
            B[i][j] = (double)(rand() % 100) / 10.0;
        }

    // Serial multiplication
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matmul_serial(A, B, C_serial);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_serial = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // OpenACC multiplication
    clock_gettime(CLOCK_MONOTONIC, &t0);
    matmul_acc(A, B, C_acc);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double t_acc = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    // Verify results
    double max_err = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double err = fabs(C_serial[i][j] - C_acc[i][j]);
            if (err > max_err) max_err = err;
        }

    printf("Matrix size     : %d x %d\n", N, N);
    printf("Serial time     : %.4f s\n", t_serial);
    printf("OpenACC time    : %.4f s\n", t_acc);
    printf("Speedup         : %.2fx\n", t_serial / t_acc);
    printf("Max error       : %e\n", max_err);
    printf("Results match   : %s\n", max_err < TOL ? "YES" : "NO");

    return 0;
}
