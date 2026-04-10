#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100
#define PROBZERO 33

int main() {
    int a[N][N]; // cities X localities
    int x[N],    // sum of infected people for all the localities of a city
        y[N];    // number of localities in a city

    srand(time(NULL));

    // FIX 1: Removed "#pragma acc parallel" — rand() is host-only / not thread-safe.
    // FIX 2: Added y[ii] = 0 initialization (was missing).
    for (int ii = 0; ii < N; ++ii) {
        x[ii] = 0;
        y[ii] = 0;
        for (int jj = 0; jj < N; ++jj) {
            a[ii][jj] = rand() % N;
            if (rand() % 100 < PROBZERO) a[ii][jj] = 0;
        }
    }

    // FIX 3: Added data directive for host <-> device data management.
    #pragma acc data copyin(a[0:N][0:N]) copy(x[0:N], y[0:N])
    {
        // FIX 4: Added "loop" keyword (was "parallel" without "loop").
        // FIX 5: Removed "collapse(2)" — it would cause race conditions on x[ii]/y[ii]
        //         since multiple jj-threads with the same ii write to the same element.
        // FIX 6: Removed incomplete "reduction" (missing operator & var list).
        //         Not needed here: each outer-loop iteration works on independent x[ii], y[ii].
        #pragma acc parallel loop
        for (int ii = 0; ii < N; ++ii) {
            for (int jj = 0; jj < N; ++jj) {
                x[ii] += a[ii][jj];
                if (a[ii][jj] > 0) y[ii]++;
            }
        }
    }

    // FIX 7: Guard against division by zero when y[ii] == 0.
    for (int ii = 0; ii < N; ++ii) {
        if (y[ii] > 0)
            printf("%.0f ", x[ii] * 100.0 / (y[ii] * N));
        else
            printf("0 ");
    }
    printf("\n");

    return 0;
}
