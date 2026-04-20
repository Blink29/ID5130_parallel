/* QR Decomposition using Modified Gram-Schmidt - OpenACC */
/* A[M][N] = Q[M][N] * R[N][N],  M > N                   */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>

#ifdef _OPENACC
#include <openacc.h>
#endif

#define M 1500
#define N 1000

double get_wall_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
  printf("QR Decomposition - Modified Gram-Schmidt (OpenACC)\n");
  printf("M = %d, N = %d\n\n", M, N);

  /* Column-major storage: A[i + j*M] = A(i,j) */
  double *A = new double[(size_t)M * N];
  double *Q = new double[(size_t)M * N];
  double *R = new double[(size_t)N * N];

  /* Initialise A with random values */
  srand(42);
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < M; ++i)
      A[i + j * M] = (double)rand() / RAND_MAX * 2.0 - 1.0;

  /* Q = copy(A), R = zeros */
  memcpy(Q, A, (size_t)M * N * sizeof(double));
  memset(R, 0,  (size_t)N * N * sizeof(double));

  double t0 = get_wall_time();

  /* Modified Gram-Schmidt with OpenACC
   *
   * Optimisations:
   *   - Single data region: Q and R copied to GPU once at the start
   *     and back once at the end, avoiding per-iteration transfers.
   *   - Column-major layout ensures coalesced memory access on GPU.
   *   - Norm and dot-product use parallel loop with reduction.
   *   - Column scaling and update use parallel loops.
   */
  #pragma acc data copy(Q[0:M*N], R[0:N*N])
  {
    for (int i = 0; i < N; ++i) {

      /* R[i][i] = ||Q[:,i]|| */
      double norm = 0.0;
      #pragma acc parallel loop reduction(+:norm) present(Q)
      for (int k = 0; k < M; ++k)
        norm += Q[k + i * M] * Q[k + i * M];

      double rii = sqrt(norm);

      #pragma acc serial present(R)
      { R[i + i * N] = rii; }

      /* Q[:,i] = Q[:,i] / R[i][i] */
      #pragma acc parallel loop present(Q)
      for (int k = 0; k < M; ++k)
        Q[k + i * M] /= rii;

      /* Orthogonalise remaining columns */
      for (int j = i + 1; j < N; ++j) {

        /* R[i][j] = Q[:,j]^T * Q[:,i] */
        double dot = 0.0;
        #pragma acc parallel loop reduction(+:dot) present(Q)
        for (int k = 0; k < M; ++k)
          dot += Q[k + j * M] * Q[k + i * M];

        #pragma acc serial present(R)
        { R[i + j * N] = dot; }

        /* Q[:,j] = Q[:,j] - R[i][j] * Q[:,i] */
        #pragma acc parallel loop present(Q)
        for (int k = 0; k < M; ++k)
          Q[k + j * M] -= dot * Q[k + i * M];
      }
    }
  }

  double elapsed = get_wall_time() - t0;
  printf("Time: %.4f s\n", elapsed);

  delete[] A;
  delete[] Q;
  delete[] R;
  return 0;
}
