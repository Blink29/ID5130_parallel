#include <iostream>
#include <omp.h>

using namespace std;

const int N = 4;
const int M = 4;

int main() {
    int a[N][M], b[N][M], c[N][M];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            a[i][j] = i + j;
            b[i][j] = i * j;
        }
    }
    
    int threads[] = {2, 4, 8};

    for (int p : threads) {
        double start = omp_get_wtime();
        #pragma omp parallel for num_threads(p)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        } 
        double end = omp_get_wtime();

        cout << "\nResult Matrix (p = " << p << "):\n";
        cout << "Time taken: " << (end - start) << " seconds\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                cout << c[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    return 0;
}