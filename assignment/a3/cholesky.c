/* Cholesky decomposition - Serial and OpenACC Parallel */
/* Decomposes symmetric positive definite matrix A = L * L^T */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#ifdef _OPENACC
#include <openacc.h>
#endif

#define TYPE       float
#define SMALLVALUE 0.001

void init(TYPE *mat, int n)
{
  for (int i = 0; i < n * n; i++) mat[i] = 0.0f;

  for (int ii = 0; ii < n; ++ii)
    for (int jj = 0; jj < n && jj < ii; ++jj) {
      mat[ii*n + jj] = (ii + jj) / (float)n / n;
      mat[jj*n + ii] = (ii + jj) / (float)n / n;
    }

  for (int ii = 0; ii < n; ++ii)
    mat[ii*n + ii] = 1.0;
}

void printMat(TYPE *a, int n)
{
  for (int ii = 0; ii < n; ++ii) {
    for (int jj = 0; jj <= ii; ++jj)
      printf("%10.6f ", a[ii*n + jj]);
    printf("\n");
  }
}

/* Serial Cholesky - original row-based formulation */
void cholesky_serial(TYPE *a, int n)
{
  for (int ii = 0; ii < n; ++ii) {
    for (int jj = 0; jj < ii; ++jj) {
      for (int kk = 0; kk < jj; ++kk)
        a[ii*n + jj] += -a[ii*n + kk] * a[jj*n + kk];
      a[ii*n + jj] /= (a[jj*n + jj] > SMALLVALUE ? a[jj*n + jj] : 1);
    }
    for (int kk = 0; kk < ii; ++kk)
      a[ii*n + ii] += -a[ii*n + kk] * a[ii*n + kk];
    a[ii*n + ii] = sqrt(a[ii*n + ii]);
  }
}

/*
 * Parallel Cholesky with OpenACC - column-based formulation
 *
 * Restructured to column-major order so that all off-diagonal
 * elements in a column can be computed independently.
 *
 * Optimizations:
 *   - Data transfer: single #pragma acc data copy around entire
 *     computation, so data is only transferred to/from the GPU once.
 *   - Parallelism: off-diagonal elements in each column are independent,
 *     parallelized with #pragma acc parallel loop.
 *   - num_gangs: adapted per column based on available iterations,
 *     capped at 128 to balance work distribution vs launch overhead.
 */
void cholesky_parallel(TYPE * restrict a, int n)
{
  #pragma acc data copy(a[0:n*n])
  {
    for (int jj = 0; jj < n; ++jj) {
      /* Diagonal element - serial on device (depends on prior columns) */
      #pragma acc serial present(a)
      {
        for (int kk = 0; kk < jj; ++kk)
          a[jj*n + jj] -= a[jj*n + kk] * a[jj*n + kk];
        a[jj*n + jj] = sqrtf(a[jj*n + jj]);
      }

      /* Off-diagonal elements - parallel across rows */
      int remaining = n - jj - 1;
      int ngangs = remaining > 128 ? 128 : (remaining > 0 ? remaining : 1);

      #pragma acc parallel loop present(a) num_gangs(ngangs)
      for (int ii = jj + 1; ii < n; ++ii) {
        TYPE s = 0.0f;
        for (int kk = 0; kk < jj; ++kk)
          s += a[ii*n + kk] * a[jj*n + kk];
        a[ii*n + jj] = (a[ii*n + jj] - s) / (a[jj*n + jj] > SMALLVALUE ? a[jj*n + jj] : 1);
      }
    }
  }
}

double get_wall_time(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[])
{
  int n = 100;
  if (argc > 1) n = atoi(argv[1]);

  TYPE *a_serial   = (TYPE *)malloc(n * n * sizeof(TYPE));
  TYPE *a_parallel = (TYPE *)malloc(n * n * sizeof(TYPE));
  if (!a_serial || !a_parallel) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  init(a_serial, n);
  memcpy(a_parallel, a_serial, n * n * sizeof(TYPE));

  /* Warm up OpenACC runtime to exclude device init from timing */
#ifdef _OPENACC
  {
    float d = 0.0f;
    #pragma acc parallel loop reduction(+:d)
    for (int i = 0; i < 10; i++) d += (float)i;
  }
#endif

  /* Serial */
  double t0 = get_wall_time();
  cholesky_serial(a_serial, n);
  double serial_time = get_wall_time() - t0;

  /* Parallel */
  t0 = get_wall_time();
  cholesky_parallel(a_parallel, n);
  double parallel_time = get_wall_time() - t0;

  printf("N=%d, Serial=%.6f, Parallel=%.6f, Speedup=%.4f\n",
         n, serial_time, parallel_time, serial_time / parallel_time);

  /* For N <= 10: print and verify both matrices */
  if (n <= 10) {
    printf("\n=== Serial Lower Triangular Matrix L ===\n");
    printMat(a_serial, n);
    printf("\n=== Parallel Lower Triangular Matrix L ===\n");
    printMat(a_parallel, n);

    float max_diff = 0.0f;
    for (int i = 0; i < n; i++)
      for (int j = 0; j <= i; j++) {
        float d = fabsf(a_serial[i*n+j] - a_parallel[i*n+j]);
        if (d > max_diff) max_diff = d;
      }
    printf("\nMax absolute difference: %e\n", max_diff);
    printf("Verification: %s\n", max_diff < 1e-5 ? "PASS (matrices match)" : "FAIL");
  }

  free(a_serial);
  free(a_parallel);
  return 0;
}
