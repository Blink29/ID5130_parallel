#include <cstdio>
#include <cstdlib>

int main()
{
    const int N = 16;
    int a[N], b[N], c[N];

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
        c[i] = 0;
    }

    // 1. #pragma acc serial — runs on device but sequentially (1 gang, 1 worker, 1 vector)
    printf("=== acc serial ===\n");
    #pragma acc serial copyin(a[0:N], b[0:N]) copyout(c[0:N])
    {
        for (int i = 0; i < N; i++)
            c[i] = a[i] + b[i];
    }
    for (int i = 0; i < 4; i++)
        printf("c[%d] = %d\n", i, c[i]);

    // 2. num_gangs, num_workers, vector_length
    printf("\n=== num_gangs, num_workers, vector_length ===\n");
    #pragma acc parallel loop num_gangs(4) num_workers(2) vector_length(4) \
        copyin(a[0:N], b[0:N]) copyout(c[0:N])
    for (int i = 0; i < N; i++)
        c[i] = a[i] * b[i];

    for (int i = 0; i < 4; i++)
        printf("c[%d] = %d\n", i, c[i]);

    // 3. private — each thread gets its own uninitialized copy of tmp
    printf("\n=== private ===\n");
    int tmp = -1;
    #pragma acc parallel loop private(tmp) copyin(a[0:N]) copyout(c[0:N])
    for (int i = 0; i < N; i++)
    {
        tmp = a[i] * 3;   // each thread has its own tmp
        c[i] = tmp + 1;
    }
    printf("tmp on host after parallel (unchanged) = %d\n", tmp);
    for (int i = 0; i < 4; i++)
        printf("c[%d] = %d\n", i, c[i]);

    // 4. firstprivate — like private, but initialized to the host value
    printf("\n=== firstprivate ===\n");
    int offset = 100;
    #pragma acc parallel loop firstprivate(offset) copyin(a[0:N]) copyout(c[0:N])
    for (int i = 0; i < N; i++)
    {
        c[i] = a[i] + offset;  // offset starts as 100 in every thread
    }
    for (int i = 0; i < 4; i++)
        printf("c[%d] = %d\n", i, c[i]);

    // 5. if clause — conditionally offload to device
    printf("\n=== if clause ===\n");
    int use_device = 1;  // set to 0 to run on host instead

    #pragma acc parallel loop if(use_device) copyin(a[0:N], b[0:N]) copyout(c[0:N])
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i] + 1000;

    printf("use_device = %d\n", use_device);
    for (int i = 0; i < 4; i++)
        printf("c[%d] = %d\n", i, c[i]);

    // Run again with if(false) -> executes on host
    use_device = 0;
    #pragma acc parallel loop if(use_device) copyin(a[0:N], b[0:N]) copyout(c[0:N])
    for (int i = 0; i < N; i++)
        c[i] = a[i] + b[i] + 2000;

    printf("use_device = %d (runs on host)\n", use_device);
    for (int i = 0; i < 4; i++)
        printf("c[%d] = %d\n", i, c[i]);

    return 0;
}
