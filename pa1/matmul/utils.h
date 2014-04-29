/********************************************************************
* EE282 Programming Assignment 1:
* Optimization of Matrix Multiplication
*
* Updated by: mgao12    04/14/2014
********************************************************************/
#ifndef __UTILS__
#define __UTILS__

#define ABS(val) ((val) > 0 ? (val) : -(val))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define SQR(a)   ((a) * (a))
#define CUBE(a)  ((a) * (a) * (a))

void   rseed();
double error(double *mat1, double *mat2, int rows, int cols);
void   mat_init(double *mat,int rows,int cols);

#endif // __UTILS__
