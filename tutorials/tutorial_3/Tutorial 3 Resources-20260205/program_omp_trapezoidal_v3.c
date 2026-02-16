/* OpenMP parallel version of trapezoidal rule */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#ifdef _OPENMP
#include<omp.h>
#endif

#define PI 3.14159265358

double func(double x)
{
  return (1.0 + sin(x));
}

double trapezoidal_rule(int n, double a, double b)
{
  double h, x, total; 		/* will be local or PRIVATE to each thread.. */
  int i;
  int my_rank = omp_get_thread_num();
  int thread_count = omp_get_num_threads();
  int local_n;			/* local number of trapeziams to be integrated by each thread. */
  double local_a, local_b;

  h = (b-a)/n;			/* h remains whether u are in serial or parallel */

  local_n = n/thread_count;	/* local values for each thread..  */
  local_a = a + my_rank*local_n*h;
  local_b = local_a + local_n*h;
  
  total = (func(local_a) + func(local_b))/2.0;
  for(i = 1; i <= local_n-1; i++) /* sharing the work, use only local_n */
    {
      x = local_a + i*h;
      total += func(x);
    }
  total = total * h;

  return total;			/* total for each thread, private */
}


int main(int argc, char* argv[])
{
  double a, b, h, final_result;
  int i, n;
  int thread_count = 1;
  

  if (argc == 2)
    {
      thread_count = strtol(argv[1], NULL, 10);
    }
  else
    {
      printf("\n A command line argument other than name of the executable is required...exiting the program..");
      return 1;
    }

#ifdef _OPENMP
  thread_count = omp_get_num_threads();
  omp_set_num_threads(thread_count);
#endif

  n = 100;			/* number of trap.. */
  a = 0.0;
  b = PI;
  final_result = 0.0;

/* # pragma omp parallel num_threads(thread_count) */
/*   { */
/*     double my_result = 0.0; */
/*     my_result = trapezoidal_rule(n, a, b); /\* parallel call here as well *\/ */
/* # pragma omp critical */
/*     final_result += my_result;  */
/*   } */

/* # pragma omp parallel num_threads(thread_count) reduction(+: final_result) */
/*     final_result += trapezoidal_rule(n, a, b); /\* call to trapezoidal rule will be parallel *\/ */

  h = (b-a)/n;
  final_result = (func(a) + func(b))/2.0;

#pragma omp parallel for reduction(+: final_result)
  for(i = 1; i<= n-1; ++i)
    final_result += func(a+i*h);
  final_result = final_result*h;
  
  
  printf("\n The area under the curve (1+sin(x)) between 0 to PI is equal to %lf \n\n", final_result);

  return 0;  
}
