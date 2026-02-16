#include <iostream>
#include <cmath>

constexpr double I_EXACT = 0.198557;

double f(double x) {
    return sin(x) / (2.0 * x * x * x);
}

double trapezoidal_rule(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = (f(a) + f(b)) / 2.0;
    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }
    return sum * h;
}

int main() {
    double a = 1.0;
    double b = M_PI;
    int n = 32;
    double result = trapezoidal_rule(f, a, b, n);
    double error = fabs(I_EXACT - result);
    printf("Computed integral: %.6f\n", result);
    printf("Exact value:       %.6f\n", I_EXACT);
    printf("Error:             %.6e\n", error);
    return 0;
}