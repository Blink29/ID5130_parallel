// MPI Collective Functions Demo: Gatherv, Scatterv, Alltoall, Wtime, Wtick
//
// Compile: mpic++ -O2 q2.cpp -o q2
// Run:     mpirun -np 4 ./q2

#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. MPI_Wtime and MPI_Wtick
    if (rank == 0) {
        cout << "  MPI_Wtime and MPI_Wtick Demo" << endl;

        double tick = MPI_Wtick();
        cout << "Timer resolution (MPI_Wtick): " << tick << " seconds" << endl;

        double t1 = MPI_Wtime();
        // Simulate some work
        double dummy = 0.0;
        for (int i = 0; i < 10000000; i++)
            dummy += i * 0.001;
        double t2 = MPI_Wtime();

        cout << "Time for 10M iterations: " << (t2 - t1) << " seconds" << endl;
        cout << "(dummy = " << dummy << ")" << endl;
        cout << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 2. MPI_Scatterv Demo
    //    Root scatters VARYING amounts of data to each process
    //    Rank i receives (i+1) elements
    if (rank == 0) {
        cout << "  MPI_Scatterv Demo" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Each rank receives (rank + 1) elements
    int recv_count = rank + 1;

    // Compute send counts and displacements on all ranks (needed for Scatterv)
    vector<int> scatterv_counts(size), scatterv_displs(size);
    int total_send = 0;
    for (int i = 0; i < size; i++) {
        scatterv_counts[i] = i + 1;       // rank i gets (i+1) elements
        scatterv_displs[i] = total_send;
        total_send += scatterv_counts[i];
    }

    // Root prepares the data
    vector<int> send_buf;
    if (rank == 0) {
        send_buf.resize(total_send);
        for (int i = 0; i < total_send; i++)
            send_buf[i] = 100 + i;

        cout << "Root sends " << total_send << " elements total: [";
        for (int i = 0; i < total_send; i++) cout << send_buf[i] << " ";
        cout << "]" << endl;
        cout << "Counts per rank: [";
        for (int c : scatterv_counts) cout << c << " ";
        cout << "]" << endl;
    }

    vector<int> scatterv_recv(recv_count);
    MPI_Scatterv(rank == 0 ? send_buf.data() : nullptr,
                 scatterv_counts.data(), scatterv_displs.data(), MPI_INT,
                 scatterv_recv.data(), recv_count, MPI_INT,
                 0, MPI_COMM_WORLD);

    // Print received data in rank order
    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            cout << "  Rank " << rank << " received " << recv_count << " elements: [";
            for (int v : scatterv_recv) cout << v << " ";
            cout << "]" << endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) cout << endl;


    // 3. MPI_Gatherv Demo
    //    Each process sends (rank+1) elements back to root
    if (rank == 0) {
        cout << "  MPI_Gatherv Demo" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Each rank prepares its own data: rank sends (rank+1) values
    int send_count = rank + 1;
    vector<int> gatherv_send(send_count);
    for (int i = 0; i < send_count; i++)
        gatherv_send[i] = rank * 10 + i;

    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            cout << "  Rank " << rank << " sends " << send_count << " elements: [";
            for (int v : gatherv_send) cout << v << " ";
            cout << "]" << endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather counts and displacements (same as scatterv_counts/displs)
    vector<int> gatherv_recv;
    if (rank == 0) gatherv_recv.resize(total_send);

    MPI_Gatherv(gatherv_send.data(), send_count, MPI_INT,
                rank == 0 ? gatherv_recv.data() : nullptr,
                scatterv_counts.data(), scatterv_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Root gathered " << total_send << " elements: [";
        for (int v : gatherv_recv) cout << v << " ";
        cout << "]" << endl;
        cout << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 4. MPI_Alltoall Demo
    //    Each process sends one int to every other process
    if (rank == 0) {
        cout << "  MPI_Alltoall Demo" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Each rank prepares 'size' values: send_buf[j] = rank*100 + j
    vector<int> alltoall_send(size), alltoall_recv(size);
    for (int j = 0; j < size; j++)
        alltoall_send[j] = rank * 100 + j;

    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            cout << "  Rank " << rank << " sends: [";
            for (int v : alltoall_send) cout << v << " ";
            cout << "]" << endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Alltoall(alltoall_send.data(), 1, MPI_INT,
                 alltoall_recv.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            cout << "  Rank " << rank << " received: [";
            for (int v : alltoall_recv) cout << v << " ";
            cout << "]" << endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // 5. Timing a collective operation with MPI_Wtime
    if (rank == 0) {
        cout << endl;
        cout << "  Timing MPI_Alltoall with MPI_Wtime" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int big_n = 100000;
    vector<int> big_send(big_n * size), big_recv(big_n * size);
    for (int i = 0; i < big_n * size; i++)
        big_send[i] = rank + i;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    MPI_Alltoall(big_send.data(), big_n, MPI_INT,
                 big_recv.data(), big_n, MPI_INT,
                 MPI_COMM_WORLD);

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Alltoall of " << big_n << " ints per rank (" << size << " ranks)" << endl;
        cout << "Max time across ranks: " << max_time << " seconds" << endl;
        cout << "Data moved per rank: " << big_n * size * sizeof(int) / 1024.0 << " KB" << endl;
    }

    MPI_Finalize();
    return 0;
}
