/* Serial version of trapezoidal rule */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define PI 3.14159265358

double func(double x)
{
  return (1.0 + sin(x));
}

void trapezoidal_rule(int n, double a, double b, double *result)
{
  double h, x, total;
  int i;

  h = (b-a)/n;			/* interval, width of the trapeziams */

  total = (func(a) + func(b))/2.0;
  for(i = 1; i <= n-1; i++) 	/* i will be private for each thread */
    total += func(a + i*h);
  total = total * h;

  *result = total; 
}


int main(void)
{
  double a, b, final_result;
  int n;

  n = 124;			/* number of trapeziams */
  a = 0.0;			/* left boundary */
  b = PI;			/* upper limit, right boundary */
  final_result = 0.0;

  trapezoidal_rule(n, a, b, &final_result);

  printf("\n The area under the curve (1+sin(x)) between 0 to PI is equal to %lf \n\n", final_result);

  return 0;  
}
