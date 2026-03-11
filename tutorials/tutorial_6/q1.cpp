#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "Demonstrating MPI Collective Operations with " << size << " ranks.\n\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 1. MPI_Reduce: Sum all ranks to Rank 0
    int reduce_send = rank;
    int reduce_recv = 0;
    MPI_Reduce(&reduce_send, &reduce_recv, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "--- MPI_Reduce ---\n";
        cout << "Summing all rank numbers to Rank 0. Result: " << reduce_recv << "\n\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 2. MPI_Allreduce: Sum all ranks to all processes
    int allreduce_send = rank;
    int allreduce_recv = 0;
    MPI_Allreduce(&allreduce_send, &allreduce_recv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) cout << "--- MPI_Allreduce ---\n";
    for (int i = 0; i < size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == i) {
            cout << "Rank " << rank << " received sum: " << allreduce_recv << "\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) cout << "\n";

    // 3. MPI_Scatter: Distribute array from Rank 0 to others
    vector<int> scatter_send;
    if (rank == 0) {
        scatter_send.resize(size);
        for (int i = 0; i < size; ++i) scatter_send[i] = i * 10;

        cout << "--- MPI_Scatter ---\n";
        cout << "Rank 0 scattering array: [";
        for (int x : scatter_send) cout << x << " ";
        cout << "]\n";
    }

    int scatter_recv = 0;
    MPI_Scatter(rank == 0 ? scatter_send.data() : nullptr, 1, MPI_INT,
                &scatter_recv, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == i) {
            cout << "Rank " << rank << " received: " << scatter_recv << "\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) cout << "\n";

    // 4. MPI_Gather: Collect values from all ranks to Rank 0
    int gather_send = rank * 100;
    vector<int> gather_recv;
    if (rank == 0) gather_recv.resize(size);

    MPI_Gather(&gather_send, 1, MPI_INT,
               rank == 0 ? gather_recv.data() : nullptr, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "--- MPI_Gather ---\n";
        cout << "Rank 0 gathered values: [";
        for (int x : gather_recv) cout << x << " ";
        cout << "]\n\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 5. MPI_Allgather: Collect values from all ranks to all ranks
    int allgather_send = rank;
    vector<int> allgather_recv(size);

    MPI_Allgather(&allgather_send, 1, MPI_INT,
                  allgather_recv.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    if (rank == 0) cout << "--- MPI_Allgather ---\n";
    for (int i = 0; i < size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == i) {
            cout << "Rank " << rank << " gathered: [";
            for (int x : allgather_recv) cout << x << " ";
            cout << "]\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) cout << "\n";

    MPI_Finalize();
    return 0;
}