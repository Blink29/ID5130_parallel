#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define PI 3.14159265358

double func(double x)
{
    return (1.0 + sin(x));
}

double trapezoidal_rule(int local_n, double local_a, double local_b)
{
    double h, x, total;
    int i;

    h = (local_b - local_a) / local_n;

    total = (func(local_a) + func(local_b)) / 2.0;
    for (i = 1; i <= local_n - 1; i++)
    {
        x = local_a + i * h;
        total += func(x);
    }
    total = total * h;

    return total;
}

int main(int argc, char** argv)
{
    int myid, size, tag = 100;
    int n, local_n;
    double a, b, h;
    double local_a, local_b, local_result;
    double final_result = 0.0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* Global parameters (same on all processes) */
    n = 124;    /* total number of trapezoids */
    a = 0.0;    /* left boundary */
    b = PI;     /* right boundary */
    h = (b - a) / n;

    /* Each process computes its local portion */
    local_n = n / size;
    local_a = a + myid * local_n * h;
    local_b = local_a + local_n * h;

    /* Each process computes its local integral */
    local_result = trapezoidal_rule(local_n, local_a, local_b);

    printf("Process %d: local_a = %.4f, local_b = %.4f, local_n = %d, local_result = %.6f\n",
           myid, local_a, local_b, local_n, local_result);

    if (myid != 0)
    {
        /* Non-root processes send their local result to process 0 */
        MPI_Send(&local_result, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }
    else
    {
        /* Process 0 sums up all the local results */
        final_result = local_result;  /* start with its own result */
        double recv_result;
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(&recv_result, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
            final_result += recv_result;
        }
        printf("\nThe area under the curve (1+sin(x)) between 0 to PI is equal to %lf\n\n", final_result);
    }

    MPI_Finalize();
    return 0;
}
