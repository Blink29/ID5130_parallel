#include <iostream>
#include <omp.h>
#include <cmath>
#include <vector>
#include <iomanip>

using namespace std;

constexpr double I_EXACT = 0.198557;

double func(double x) {
    return sin(x) / (2.0 * x * x * x);
}

void solve(int n, int p) {
    double a = 1.0;
    double b = M_PI;
    double h = (b - a) / n;
    
    double sum = func(a) + func(b);
    
    #pragma omp parallel for num_threads(p) reduction(+:sum) 
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum += 2.0 * func(x);
        } else {
            sum += 4.0 * func(x);
        }
    }
    
    double integral = (h / 3.0) * sum;
    double error = abs(I_EXACT - integral);
    
    cout << "N = " << setw(3) << n 
              << ", P = " << p 
              << ", Integral = " << fixed << setprecision(6) << integral 
              << ", Error = " << scientific << error << endl;
}

int main() {
    vector<int> n_values = {32, 128, 256};
    vector<int> p_values = {2, 4};
    
    cout << "Simpson's Rule Integration Results:" << endl;
    cout << "-----------------------------------" << endl;

    for (int n : n_values) {
        for (int p : p_values) {
            solve(n, p);
        }
    }
    
    return 0;
}