#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv)
{
    int myid, size;
    int N;
    int local_start, local_end, local_n;
    int *is_composite;  /* 1 = composite, 0 = prime */
    int prime, local_count, global_count;
    int i, first_multiple;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    N = 100;  /* find all primes up to N */

    /* Block decomposition: divide range [2, N] among processes */
    int range = N - 1;  /* numbers from 2 to N */
    local_n = range / size;
    int remainder = range % size;

    /* Distribute remainder evenly: first 'remainder' processes get one extra */
    if (myid < remainder)
    {
        local_n += 1;
        local_start = 2 + myid * local_n;
    }
    else
    {
        local_start = 2 + remainder * (local_n + 1) + (myid - remainder) * local_n;
    }
    local_end = local_start + local_n - 1;

    /* Allocate local sieve array */
    is_composite = (int*)calloc(local_n, sizeof(int));  /* initialized to 0 (all prime) */

    if (myid == 0)
        printf("Sieve of Eratosthenes up to %d using %d processes\n\n", N, size);

    /* Sieve: iterate over primes up to sqrt(N) */
    int sqrt_N = (int)sqrt((double)N);
    prime = 2;

    while (prime <= sqrt_N)
    {
        /* Mark multiples of 'prime' in local range */
        /* Find the first multiple of prime >= local_start */
        if (prime * prime > local_start)
            first_multiple = prime * prime;
        else
        {
            /* Find smallest multiple of prime >= local_start */
            first_multiple = local_start + (prime - (local_start % prime)) % prime;
        }

        /* Mark all multiples of prime in local range */
        for (i = first_multiple; i <= local_end; i += prime)
        {
            is_composite[i - local_start] = 1;  /* mark as composite */
        }

        /* Process 0 finds the next prime */
        if (myid == 0)
        {
            /* Find next unmarked number after current prime */
            int next = prime + 1;
            while (next <= local_end && is_composite[next - local_start] == 1)
                next++;
            prime = next;
        }

        /* Broadcast the next prime to all processes */
        MPI_Bcast(&prime, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    /* Count local primes */
    local_count = 0;
    for (i = 0; i < local_n; i++)
    {
        if (is_composite[i] == 0)
            local_count++;
    }

    /* Print local primes (for small N, useful for verification) */
    /* Each process prints in order using a barrier */
    for (int p = 0; p < size; p++)
    {
        if (myid == p)
        {
            printf("Process %d [%d-%d]: ", myid, local_start, local_end);
            for (i = 0; i < local_n; i++)
            {
                if (is_composite[i] == 0)
                    printf("%d ", local_start + i);
            }
            printf(" (%d primes)\n", local_count);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Reduce to get total prime count */
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0)
        printf("\nTotal number of primes up to %d: %d\n", N, global_count);

    free(is_composite);
    MPI_Finalize();
    return 0;
}
