#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 15

// f(x) = sin(5x),  f'(x) = 5*cos(5x)
double f(double x)  { return sin(5.0 * x); }
double fp(double x) { return 5.0 * cos(5.0 * x); }

// TDMA (Thomas algorithm) for tridiagonal system
// a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
void tdma(int n, double *a, double *b, double *c, double *d, double *x) {
    double *cp = (double *)malloc(n * sizeof(double));
    double *dp = (double *)malloc(n * sizeof(double));

    // Forward sweep
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for (int i = 1; i < n; i++) {
        double m = b[i] - a[i] * cp[i - 1];
        cp[i] = c[i] / m;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }

    // Back substitution
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; i--)
        x[i] = dp[i] - cp[i] * x[i + 1];

    free(cp);
    free(dp);
}

int main() {
    double xp[N], fv[N], exact[N];
    double pade[N];   // Pade derivative
    double expl[N];   // Explicit (CDS2 + FD1/BD1) derivative

    double x0 = 0.0, xn = 3.0;
    double h = (xn - x0) / (N - 1);

    // Evaluate function and exact derivative
    for (int i = 0; i < N; i++) {
        xp[i] = x0 + i * h;
        fv[i] = f(xp[i]);
        exact[i] = fp(xp[i]);
    }

    // Method 1: Explicit schemes — CDS2 interior, FD1/BD1 boundary
    expl[0] = (fv[1] - fv[0]) / h;                        // FD1
    for (int i = 1; i < N - 1; i++)
        expl[i] = (fv[i + 1] - fv[i - 1]) / (2.0 * h);   // CDS2
    expl[N - 1] = (fv[N - 1] - fv[N - 2]) / h;            // BD1

    // Method 2: 4th-order Pade scheme (interior) +
    //           3rd-order compact boundary schemes
    //
    //  Interior (i=1..N-2):
    //    (1/4) f'_{i-1} + f'_i + (1/4) f'_{i+1} = (3/2)(f_{i+1} - f_{i-1})/(2h)
    //
    //  Left boundary (i=0):
    //    f'_0 + 2 f'_1 = (-5 f_0 + 4 f_1 + f_2) / (2h)
    //
    //  Right boundary (i=N-1):
    //    2 f'_{N-2} + f'_{N-1} = (-f_{N-3} - 4 f_{N-2} + 5 f_{N-1}) / (2h)
    double a[N], b[N], c[N], d[N];

    // Left boundary: f'_0 + 2 f'_1 = (-5f_0 + 4f_1 + f_2) / (2h)
    a[0] = 0.0;
    b[0] = 1.0;
    c[0] = 2.0;
    d[0] = (-5.0 * fv[0] + 4.0 * fv[1] + fv[2]) / (2.0 * h);

    // Interior: (1/4) f'_{i-1} + f'_i + (1/4) f'_{i+1} = (3/2)(f_{i+1} - f_{i-1})/(2h)
    for (int i = 1; i < N - 1; i++) {
        a[i] = 0.25;
        b[i] = 1.0;
        c[i] = 0.25;
        d[i] = 1.5 * (fv[i + 1] - fv[i - 1]) / (2.0 * h);
    }

    // Right boundary: 2 f'_{N-2} + f'_{N-1} = (-f_{N-3} - 4f_{N-2} + 5f_{N-1}) / (2h)
    a[N - 1] = 2.0;
    b[N - 1] = 1.0;
    c[N - 1] = 0.0;
    d[N - 1] = (-fv[N - 3] - 4.0 * fv[N - 2] + 5.0 * fv[N - 1]) / (2.0 * h);

    // Solve tridiagonal system (serial TDMA)
    tdma(N, a, b, c, d, pade);

    // Output results
    printf("%12s %12s %12s %12s\n", "x", "Exact", "Pade4", "CDS2");
    printf("-----------------------------------------------------\n");
    for (int i = 0; i < N; i++)
        printf("%12.6f %12.6f %12.6f %12.6f\n", xp[i], exact[i], pade[i], expl[i]);

    // Write to file for plotting
    FILE *fp_out = fopen("q3_results.dat", "w");
    fprintf(fp_out, "# x  exact  pade4  cds2\n");
    for (int i = 0; i < N; i++)
        fprintf(fp_out, "%f %f %f %f\n", xp[i], exact[i], pade[i], expl[i]);
    fclose(fp_out);
    printf("\nResults written to q3_results.dat\n");

    return 0;
}
