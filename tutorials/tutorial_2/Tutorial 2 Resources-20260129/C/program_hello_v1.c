/* Hello world program using OpenMP */
#include<stdio.h>
#include<omp.h>
void main()
{

  printf("\n About to enter the parallel world...");

#pragma omp parallel
  printf("\n Hello world from thread number %d", omp_get_thread_num());

  printf("\n Entered the serial world...\n");
  
}

