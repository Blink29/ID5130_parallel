#include <mpi.h>
#include <iostream>

using namespace std;

// Define the struct
struct Payload {
    double a[3];
    double b[2];
    int n;
};

// Function to create the custom MPI datatype for the struct
MPI_Datatype create_payload_type() {
    MPI_Datatype payload_type;

    int block_lengths[3] = {3, 2, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};

    // Calculate memory displacements
    Payload sample;
    MPI_Aint base, addr_a, addr_b, addr_n;

    MPI_Get_address(&sample, &base);
    MPI_Get_address(&sample.a, &addr_a);
    MPI_Get_address(&sample.b, &addr_b);
    MPI_Get_address(&sample.n, &addr_n);

    displacements[0] = addr_a - base;
    displacements[1] = addr_b - base;
    displacements[2] = addr_n - base;

    // Create and commit the type
    MPI_Type_create_struct(3, block_lengths, displacements, types, &payload_type);
    MPI_Type_commit(&payload_type);
    
    return payload_type;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Payload data; // Declare payload
    
    // Rank 0 initializes data
    if (rank == 0) {
        data.a[0] = 12.67;
        data.a[1] = 56.45;
        data.a[2] = 25.32;
        
        data.b[0] = 56.79;
        data.b[1] = 98.26;
        
        data.n = 10;
        
        cout << "Broadcasting Payload array from Rank 0." << endl << endl;
    }

    // Creating derived datatype on all processes
    MPI_Datatype payload_type = create_payload_type();

    // Broadcast the payload
    MPI_Bcast(&data, 1, payload_type, 0, MPI_COMM_WORLD);

    // Barrier just to ensure safe sequential console printing
    for (int r = 0; r < size; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == r) {
            cout << "Rank " << rank << " received data:" << endl;
            cout << "  a[] = { " << data.a[0] << ", " << data.a[1] << ", " << data.a[2] << " }" << endl;
            cout << "  b[] = { " << data.b[0] << ", " << data.b[1] << " }" << endl;
            cout << "  n   = " << data.n << endl << endl;
        }
    }

    // Clean up
    MPI_Type_free(&payload_type);
    MPI_Finalize();
    return 0;
}
