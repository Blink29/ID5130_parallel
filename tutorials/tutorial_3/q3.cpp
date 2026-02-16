#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Serial sieve
vector<uint8_t> sieve_serial(int n) {
    vector<uint8_t> is_prime(n + 1, 1);
    if (n >= 0) is_prime[0] = 0;
    if (n >= 1) is_prime[1] = 0;

    for (int k = 2; 1LL * k * k <= n; ++k) {
        if (is_prime[k]) {
            for (long long i = 1LL * k * k; i <= n; i += k) {
                is_prime[(size_t)i] = 0;
            }
        }
    }
    return is_prime;
}

// OpenMP "try" sieve: outer k loop serial, inner marking parallel
vector<uint8_t> sieve_omp(int n, int p) {
    vector<uint8_t> is_prime(n + 1, 1);
    if (n >= 0) is_prime[0] = 0;
    if (n >= 1) is_prime[1] = 0;

#ifdef _OPENMP
    omp_set_num_threads(p);
#endif

    for (int k = 2; 1LL * k * k <= n; ++k) {
        if (is_prime[k]) {
            // Mark multiples of k in parallel
            #pragma omp parallel for
            for (long long i = k * k; i <= n; i += k) {
                is_prime[i] = 0;
}
        }
    }
    return is_prime;
}

static long long count_primes(const vector<uint8_t>& is_prime) {
    long long cnt = 0;
    for (size_t i = 2; i < is_prime.size(); ++i) cnt += is_prime[i] ? 1 : 0;
    return cnt;
}

int main() {
    int n = 1000000000; // pick large n to test (adjust to your RAM/time)
    cout << "n = " << n << "\n";

#ifdef _OPENMP
    auto now = [](){ return omp_get_wtime(); };
#else
    auto now = [](){ return 0.0; };
#endif

    // Serial
    double t0 = now();
    auto prime_s = sieve_serial(n);
    double t1 = now();
    cout << "[Serial] primes = " << count_primes(prime_s)
              << " | time = " << (t1 - t0) << " s\n";

    // OpenMP p=2,4
    for (int p : {2, 4}) {
        double a0 = now();
        auto prime_p = sieve_omp(n, p);
        double a1 = now();

        cout << "[OpenMP p=" << p << "] primes = " << count_primes(prime_p)
                  << " | time = " << (a1 - a0) << " s\n";
    }

    return 0;
}
