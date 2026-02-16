#include <iostream>
#include <omp.h>

using namespace std;

double** allocate_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void populate_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

void matrix_add(double** a, double** b, double** c, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

void matrix_mult(double** a, double** b, double** c, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

int main() {
    srand(time(NULL));
    int sizes[] = {50, 100}; 
    int num_sizes = 2;
    int thread_counts[] = {2, 4, 8}; 
    int num_thread_tests = 3;

    printf("Matrix Operation Performance Test using OpenMP\n");
    printf("----------------------------------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        printf("\n>>> SIZE N = %d\n", n);

        // Allocation
        double** A = allocate_matrix(n);
        double** B = allocate_matrix(n);
        double** C = allocate_matrix(n);

        populate_matrix(A, n);
        populate_matrix(B, n);

        // 1. Baseline: Run with 1 thread (Sequential) for comparison
        omp_set_num_threads(1);
        double start = omp_get_wtime();
        matrix_add(A, B, C, n);
        double end = omp_get_wtime();
        printf("  [Threads: 1] Addition:       %f sec (Baseline)\n", end - start);

        start = omp_get_wtime();
        matrix_mult(A, B, C, n);
        end = omp_get_wtime();
        printf("  [Threads: 1] Multiplication: %f sec (Baseline)\n", end - start);
        printf("  ------------------------------------------\n");

        // 2. Multi-threaded tests
        for (int t = 0; t < num_thread_tests; t++) {
            int threads = thread_counts[t];
            omp_set_num_threads(threads);

            // Addition Test
            start = omp_get_wtime();
            matrix_add(A, B, C, n);
            end = omp_get_wtime();
            printf("  [Threads: %d] Addition:       %f sec\n", threads, end - start);

            // Multiplication Test
            start = omp_get_wtime();
            matrix_mult(A, B, C, n);
            end = omp_get_wtime();
            printf("  [Threads: %d] Multiplication: %f sec\n", threads, end - start);
        }

        // Cleanup
        free_matrix(A, n);
        free_matrix(B, n);
        free_matrix(C, n);
    }

    return 0;
}