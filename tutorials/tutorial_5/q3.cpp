#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv)
{
    int myid, size, tag = 100;
    double value, sum = 0.0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid != 0)
    {
        /* Each non-host process sends its rank as a float value */
        value = (double)myid;
        printf("Process %d sending value: %.2f\n", myid, value);
        MPI_Send(&value, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }
    else
    {
        /* Host process (rank 0) receives from all others and sums */
        sum = 0.0;
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(&value, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
            printf("Host received %.2f from process %d\n", value, i);
            sum += value;
        }
        printf("\nFinal sum = %.2f\n", sum);
    }

    MPI_Finalize();
    return 0;
}
