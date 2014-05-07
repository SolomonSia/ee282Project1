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

#include <xmmintrin.h>

#define NUM_THREADS 4
#define BLOCK_SIZE 8
#define BIG_BLOCK 32


/*****************************
* Multi-threaded
*****************************/
#if 1
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

void matThread(int dim, const double*__restrict__ A, const double*__restrict__ B, double*__restrict__ C,
		int row_begin, int row_end, int col_begin, int col_end){

    if(dim == 16) {
      int i, j, k;
      int base_i = row_begin * 16;

        for (i = 0; i < 8; i++){
          for (j = 0; j < 8; j++){
            int index = base_i + col_begin + j;
            register double c = C[index];
            for (k = 0; k < 8; k++){
              c += A[base_i + k] * B[k * 16 + j + col_begin];
            }
            C[index] = c;
          }
          base_i += 16;
        }

        base_i = row_begin * 16;

        for (i = 0; i < 8; i++){
          for (j = 0; j < 8; j++){
            int index = base_i + col_begin + j;
            register double c = C[index];
            for (k = 8; k < 16; k++){
              c += A[base_i + k] * B[k * 16 + j + col_begin];
            }
            C[index] = c;
          }
          base_i += 16;
        }
      
  }

  else if(dim == 64) {

    int k1,i0,j0,k0,i,j,k;

    for(k1 = 0; k1 < 64;k1 += 32) {

      for (i0 = row_begin; i0 < row_begin+32; i0 += 8){
        for (j0 = col_begin; j0 < col_begin+32; j0 += 8){
          for (k0 = k1; k0 < k1+32; k0 += 8){

            for (i = i0; i < i0+8; i++){
              for (j = j0; j < j0+8; j++){
                register double c = C[i*dim +j];
                for (k = k0; k < k0+8; k++){
                  c += A[i*dim + k] * B[k*dim + j];
                }
                C[i*dim + j] = c;
              }
            }
          }
        }
      }
    }

  }

  else if (dim == 256) {

    int i0, j0, k0, i1, j1, k1, i, j, k;

    for(i1=row_begin;i1<row_begin+128;i1+=32){
      for(j1=col_begin;j1<col_begin+128;j1+=32){
        for(k1=0;k1<dim;k1+=32){

          for (i0=i1; i0<i1+32; i0+=8){
            for (j0=j1; j0<j1+32; j0+=8){
              for (k0=k1; k0<k1+32; k0+=8){

                for (i = i0; i < i0+8; i++){
                  for (j = j0; j < j0+8; j++){
                    register double c = C[i*dim +j];
                    for (k = k0; k < k0+8; k++){
                      c += A[i*dim + k] * B[k*dim + j];
                    }
                    C[i*dim + j] = c;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  else if(dim == 1024) {
    int i0, j0, k0, i1, j1, k1, i, j, k;

     for(i1=row_begin;i1<row_begin+512;i1+=32){
      for(j1=col_begin;j1<col_begin+512;j1+=32){
        for(k1=0;k1<dim;k1+=32){

          for (i0=i1; i0<i1+32; i0+=8){
            for (j0=j1; j0<j1+32; j0+=8){
              for (k0=k1; k0<k1+32; k0+=8){

                for (i = i0; i < i0+8; i++){
                  for (j = j0; j < j0+8; j++){
                    register double c = C[i*dim +j];
                    for (k = k0; k < k0+8; k++){
                      c += A[i*dim + k] * B[k*dim + j];
                    }
                    C[i*dim + j] = c;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  else {
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
                    register double c = C[i*dim +j];
	                  for (k = k0; k < MIN(k0+BLOCK_SIZE, dim); k++){
                      c += A[i*dim + k] * B[k*dim + j];
	                  }
                    C[i*dim + j] = c;
	                }
	              }
	            }
	          }
          }
        }
      }
    }
  }
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

inline void *worker_func (void * __restrict__ arg) {
  thread_arg * __restrict__ targ = (thread_arg * __restrict__ )arg;

  const double * __restrict__ A = targ->A;
  const double * __restrict__ B = targ->B;
  double * __restrict__ C = targ->C;
  int dim = targ->dim;
  int row_begin = targ -> row_begin;
  int row_end = targ -> row_end;
  int col_begin = targ -> col_begin;
  int col_end = targ -> col_end;
  matThread(dim, A, B, C, row_begin, row_end, col_begin, col_end);

  return NULL;
}

/*inline void mult_4(int N, const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C) {
  int i, j, k;
  __m128d a_00 = _mm_loadu_pd(&A[0]);
  __m128d b_00 = _mm_set_pd(B[0], B[4]);
  __m128d c_00 = _mm_loadu_pd(&C[0]);
  C[0] += _mm_mul_pd(a_00, b_00) + _mm_mul_pd(a_02, b_02);
}*/


void matmul(int N, const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C) {
   
  pthread_t workers[NUM_THREADS - 1];
  thread_arg args[NUM_THREADS];
  int stripe = (N+1) / 2;
  int id;

  if(N == 4) {
    int i, j, k;
    for(i = 0 ; i < 4; i++) {
      for(j = 0; j < 4; j++) {
        register double c = C[i*4+j];
        for(k = 0; k < 4; k++) {
          c += A[i*4 + k] * B[k*4 + j];
        }
        C[i*4+j] = c;
      }
    }
  }

   else {
    args[0] = (thread_arg){ A, B, C, N, 0, stripe, 0, stripe};
    pthread_create(&workers[0], NULL, worker_func, &args[0]);

    args[1] = (thread_arg){ A, B, C, N, 0, stripe, stripe, N};
    pthread_create(&workers[1], NULL, worker_func, &args[1]);

    args[2] = (thread_arg){ A, B, C, N, stripe, N, 0, stripe};
    pthread_create(&workers[2], NULL, worker_func, &args[2]);

    args[3] = (thread_arg){ A, B, C, N, stripe, N, stripe, N};
    worker_func(&args[NUM_THREADS - 1]);
   
    for (id = 0; id < NUM_THREADS - 1; id++) {
      if (pthread_join(workers[id], NULL) != 0) {
        fprintf(stderr, "Fail to join thread!");
        exit(1);
      }
    }
  }
}

#endif
