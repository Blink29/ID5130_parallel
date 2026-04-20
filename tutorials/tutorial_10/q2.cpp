/* 1D Convolution Filter - OpenACC */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>

#ifdef _OPENACC
#include <openacc.h>
#endif

#define K 5

double get_wall_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void convolve(const int *A, const int *F, int *B, int N) {
  int half = K / 2;

  #pragma acc data copyin(A[0:N], F[0:K]) copyout(B[0:N])
  {
    #pragma acc parallel loop present(A, F, B)
    for (int i = 0; i < N; ++i) {
      int sum = 0;
      for (int m = 0; m < K; ++m) {
        int j = i - half + m;
        if (j >= 0 && j < N)
          sum += F[m] * A[j];
      }
      B[i] = sum;
    }
  }
}

/* Verify against the example from the problem statement */
void verify_example() {
  int A[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int F[] = {3, 4, 5, 4, 3};
  int expected[] = {22, 38, 57, 76, 95, 114, 106, 86};
  int N = 8;
  int B[8];

  convolve(A, F, B, N);

  printf("=== Verification with example ===\n");
  printf("Input:    ");
  for (int i = 0; i < N; ++i) printf("%d ", A[i]);
  printf("\nFilter:   ");
  for (int i = 0; i < K; ++i) printf("%d ", F[i]);
  printf("\nOutput:   ");
  for (int i = 0; i < N; ++i) printf("%d ", B[i]);
  printf("\nExpected: ");
  for (int i = 0; i < N; ++i) printf("%d ", expected[i]);

  bool pass = true;
  for (int i = 0; i < N; ++i)
    if (B[i] != expected[i]) { pass = false; break; }
  printf("\nResult:   %s\n\n", pass ? "PASS" : "FAIL");
}

int main() {
  printf("1D Convolution Filter (OpenACC), K = %d\n\n", K);

  verify_example();

  int F[K] = {3, 4, 5, 4, 3};
  int sizes[] = {1000, 2000, 3000};

  for (int s = 0; s < 3; ++s) {
    int N = sizes[s];
    int *A = new int[N];
    int *B = new int[N];

    srand(42);
    for (int i = 0; i < N; ++i)
      A[i] = rand() % 100;

    double t0 = get_wall_time();
    convolve(A, F, B, N);
    double elapsed = get_wall_time() - t0;

    printf("N = %5d | Time: %.6f s\n", N, elapsed);

    delete[] A;
    delete[] B;
  }

  return 0;
}
