#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Initialize matrix and vector
void build_problem(int n, vector<double>& matrix, vector<double>& x) {
    matrix.resize(n * n);
    x.resize(n);
    for (int j = 0; j < n; ++j) {
        x[j] = j + 1;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i * n + j] = (i + 1) + 0.1 * (j + 1);
        }
    }
}

// Compute matrix-vector multiplication serially for verification
void serial_matvec(const vector<double>& matrix, const vector<double>& x, vector<double>& y_serial, int n) {
    y_serial.resize(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            y_serial[i] += matrix[i * n + j] * x[j];
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 8; // Default matrix size
    if (argc > 1) {
        n = stoi(argv[1]); // Allow custom size
    }

    // Master process initializes data
    vector<double> matrix_global;
    vector<double> x(n);
    if (rank == 0) {
        build_problem(n, matrix_global, x);
    }

    // Broadcast vector x to all processes
    MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate rows per process
    int local_rows = n / size;
    int remainder = n % size;

    // Displacement and block size arrays for Scatterv/Gatherv
    vector<int> sendcounts(size);
    vector<int> displs(size);
    
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        int rows = local_rows + (i < remainder ? 1 : 0);
        sendcounts[i] = rows * n;
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    // Determine the number of rows for the current rank
    int my_rows = local_rows + (rank < remainder ? 1 : 0);
    
    // Allocate space for the local matrix block
    vector<double> local_matrix(my_rows * n);

    // Scatter the matrix using block-decomposition
    MPI_Scatterv(rank == 0 ? matrix_global.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_matrix.data(), my_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local matrix-vector product
    vector<double> local_y(my_rows, 0.0);
    for (int i = 0; i < my_rows; ++i) {
        for (int j = 0; j < n; ++j) {
            local_y[i] += local_matrix[i * n + j] * x[j];
        }
    }

    // Prepare arrays for gathering the result
    vector<int> recvcounts(size);
    vector<int> recvdispls(size);
    offset = 0;
    for (int i = 0; i < size; ++i) {
        int rows = local_rows + (i < remainder ? 1 : 0);
        recvcounts[i] = rows;
        recvdispls[i] = offset;
        offset += recvcounts[i];
    }

    // Gather the local results onto the master process
    vector<double> y_parallel;
    if (rank == 0) {
        y_parallel.resize(n);
    }

    MPI_Gatherv(local_y.data(), my_rows, MPI_DOUBLE,
                rank == 0 ? y_parallel.data() : nullptr, recvcounts.data(), recvdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Verification on rank 0
    if (rank == 0) {
        vector<double> y_serial;
        serial_matvec(matrix_global, x, y_serial, n);

        bool pass = true;
        for (int i = 0; i < n; ++i) {
            if (abs(y_parallel[i] - y_serial[i]) > 1e-10) {
                pass = false;
                break;
            }
        }

        cout << "Matrix-Vector Multiplication (Row Block Decomposition)" << endl;
        cout << "Matrix Size (N x N) : " << n << " x " << n << endl;
        cout << "Number of MPI ranks : " << size << endl;
        
        cout << "Results Match Serial: " << (pass ? "YES" : "NO") << endl;
        
        cout << "\nComputed Vector Y: [ ";
        for (double val : y_parallel) {
            cout << val << " ";
        }
        cout << "]" << endl;
    }

    MPI_Finalize();
    return 0;
}
