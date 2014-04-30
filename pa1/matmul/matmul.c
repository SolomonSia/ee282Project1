/********************************************************************
* EE282 Programming Assignment 1:
* Optimization of Matrix Multiplication
*
* Updated by: mgao12    04/14/2014
********************************************************************/

// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 4
#define BLOCK_SIZE 32
#define BIG_BLOCK 128
/*****************************
* Single-threaded
*****************************/
#if 0
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

typedef struct {
   const double *A;
   const double *B;
   double *C;
   int dim;
   int row_begin;
   int row_end;
   int col_begin;
   int col_end;
} thread_arg;


void matmul(int N, const double*__restrict__ A, const double*__restrict__ B, double* __restrict__ C) {

/* matrix transpose */
//   double Bt[N][N];
//   double *__restrict__ Btt = (double *)Bt;
   int i, j, k, i0, j0, k0,i1,j1,k1;
 /*  for(i=0;i<N;i+=BLOCK_SIZE){
    for(j=0;j<N;j+=BLOCK_SIZE){
     for(k=i;k<i+BLOCK_SIZE;++k){
      for(l=j;l<j+BLOCK_SIZE;++l){
        Btt[k+l*N] = B[l+k*N];
}}}}
*/

   for(i1=0;i1<N;i1+=BIG_BLOCK){
   for(j1=0;j1<N;j1+=BIG_BLOCK){
   for(k1=0;k1<N;k1+=BIG_BLOCK){

   for (i0=i1; i0<MIN(i1+BIG_BLOCK,N); i0+=BLOCK_SIZE){

       for (j0=j1; j0<MIN(j1+BIG_BLOCK,N); j0+=BLOCK_SIZE){

           for (k0=k1; k0<(MIN(k1+BIG_BLOCK,N)); k0+=BLOCK_SIZE){

               for (i = i0; i < MIN(i0+BLOCK_SIZE, N); i++){
                   for (j = j0; j < MIN(j0+BLOCK_SIZE, N); j++){
                       for (k = k0; k < MIN(k0+BLOCK_SIZE, N); k++){
                           C[i*N + j] += A[i*N + k] * B[k*N + j];
                       }
                   }
               }
       }}}

}}}

}
#endif

/*****************************
* Multi-threaded
*****************************/
#if 1
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

void matThread(int dim, const double*__restrict__ A, const double*__restrict__ B, double*__restrict__ C,
		int row_begin, int row_end, int col_begin, int col_end){

	   int i0, j0, k0,i1,j1,k1;
	   int i, j, k;

	   for(i1=row_begin;i1<row_end;i1+=BIG_BLOCK){
	   for(j1=col_begin;j1<col_end;j1+=BIG_BLOCK){
	   for(k1=0;k1<dim;k1+=BIG_BLOCK){

		   for (i0=i1; i0<MIN(i1+BIG_BLOCK,row_end); i0+=BLOCK_SIZE){
		        for (j0=j1; j0<MIN(j1+BIG_BLOCK,col_end); j0+=BLOCK_SIZE){
		            for (k0=k1; k0<MIN(k1+BIG_BLOCK,dim); k0+=BLOCK_SIZE){

		            for (i = i0; i < MIN(i0+BLOCK_SIZE, row_end); i++){
	                   for (j = j0; j < MIN(j0+BLOCK_SIZE, col_end); j++){
	                       for (k = k0; k < MIN(k0+BLOCK_SIZE, dim); k++){
	                           C[i*dim + j] += A[i*dim + k] * B[k*dim + j];
	                       }
	                   }
	               }
	           }
	       }
	   }
	   }}}
}

typedef struct {
   const double * __restrict__ A;
   const double  * __restrict__ B;
   double * __restrict__ C;
   int dim;
   int row_begin;
   int row_end;
   int col_begin;
   int col_end;
} thread_arg;

void * worker_func (void * __restrict__ arg) {
   thread_arg * __restrict__ targ = (thread_arg * __restrict__ )arg;

   const double * __restrict__ A = targ->A;
   const double * __restrict__ B = targ->B;
   double * __restrict__ C = targ->C;
   int dim = targ->dim;
   int row_begin = targ -> row_begin;
   int row_end = targ -> row_end;
   int col_begin = targ -> col_begin;
   int col_end = targ -> col_end;
   matThread(dim,A,B,C,row_begin,row_end,col_begin,col_end);

   return NULL;
}

void matmul(int N, const double* __restrict__ A, const double* __restrict__ B,
		double* __restrict__ C) {
   pthread_t workers[NUM_THREADS - 1];
   thread_arg args[NUM_THREADS];
   int stripe = (N+1) / 2;
   int i;

   args[0] = (thread_arg){ A, B, C, N, 0, stripe, 0, stripe};
   pthread_create(&workers[0], NULL, worker_func, &args[0]);

   args[1] = (thread_arg){ A, B, C, N, 0, stripe, stripe, N};
   pthread_create(&workers[1], NULL, worker_func, &args[1]);

   args[2] = (thread_arg){ A, B, C, N, stripe, N, 0, stripe};
   pthread_create(&workers[2], NULL, worker_func, &args[2]);

   args[3] = (thread_arg){ A, B, C, N, stripe, N, stripe, N};
   worker_func(&args[NUM_THREADS - 1]);

   for (i = 0; i < NUM_THREADS - 1; i++) {
       if (pthread_join(workers[i], NULL) != 0) {
           fprintf(stderr, "Fail to join thread!");
           exit(1);
       }
   }
}
#endif
