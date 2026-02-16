#include <cstdlib>
#include <cstdio>
#include <omp.h>

#ifndef N
#define N 5
#endif
#ifndef FS
#define FS 38
#endif

struct node {
    int data;
    int fibdata;
    struct node* next;
};

int fib(int n) {
    int x, y;
    if (n < 2) {
        return (n);
    } else {
        x = fib(n - 1);
        y = fib(n - 2);
        return (x + y);
    }
}

void processwork(struct node* p) {
    int n;
    n = p->data;
    p->fibdata = fib(n);
}

struct node* init_list(struct node* p) {
    int i;
    struct node* head = NULL;
    struct node* temp = NULL;

    head = (struct node*)std::malloc(sizeof(struct node));
    p = head;
    p->data = FS;
    p->fibdata = 0;
    for (i = 0; i < N; i++) {
        temp = (struct node*)std::malloc(sizeof(struct node));
        p->next = temp;
        p = temp;
        p->data = FS + i + 1;
        p->fibdata = i + 1;
    }
    p->next = NULL;
    return head;
}

int main(int argc, char* argv[]) {
    double start, end;
    struct node* p = NULL;
    struct node* temp = NULL;
    struct node* head = NULL;

    std::printf("Process linked list\n");
    std::printf("  Each linked list node will be processed by function 'processwork()'\n");
    std::printf("  Each ll node will compute %d fibonacci numbers beginning with %d\n", N, FS);

    p = init_list(p);
    head = p;
    start = omp_get_wtime();

    // start = omp_get_wtime();
    // {
    //     while (p != NULL) {
    //         processwork(p);
    //         p = p->next;
    //     }
    // }
    // end = omp_get_wtime();

    //  int count = 0;
    //  struct node* temp_p = p;
    //  while (temp_p != NULL) {
    //      count++;
    //      temp_p = temp_p->next;
    //  }

    //  struct node** list_array = (struct node**)malloc(count * sizeof(struct node*));
    //  temp_p = p;
    //  for (int i = 0; i < count; i++) {
    //      list_array[i] = temp_p;
    //      temp_p = temp_p->next;
    //  }

    //  #pragma omp parallel for schedule(dynamic)
    //  for (int i = 0; i < count; i++) {
    //      processwork(list_array[i]);
    //  }

    //  free(list_array);

    #pragma omp parallel
    {
        #pragma omp single 
        {
            node* p = head;
            while (p) {
                #pragma omp task 
                processwork(p);
            p = p->next;
            }
        }
    }

    end = omp_get_wtime();

    p = head;
    while (p != NULL) {
        std::printf("%d : %d\n", p->data, p->fibdata);
        temp = p->next;
        std::free(p);
        p = temp;
    }
    std::free(p);

    std::printf("Compute Time: %f seconds\n", end - start);

    return 0;
}
