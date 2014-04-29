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

/*****************************
 * Single-threaded
 *****************************/
#if 0
void matmul(int N, const double* __restrict__ A, const double* __restrict__
 B, double* __restrict__ C) {
    int i, j, k;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}
#endif

/*****************************
 * Multi-threaded
 *****************************/
#if 1
typedef struct {
    const double * __restrict__ A;
    const double * __restrict__ B;
    double * __restrict__ C;
    int dim;
    int row_begin;
    int row_end;
} thread_arg;

void * worker_func (void * __restrict__ arg) {
    thread_arg * __restrict__ targ = (thread_arg * __restrict__)arg;

    const double *__restrict__ A = targ->A;
    const double *__restrict__ B = targ->B;
    double *__restrict__ C = targ->C;
    int dim = targ->dim;

    int i, j, k;
    for (i = targ->row_begin; i < targ->row_end; i++) {
        for (j = 0; j < dim; j++) {
            for (k = 0; k < dim; k++) {
                C[i*dim + j] += A[i*dim + k] * B[k*dim + j];
            }
        }
    }
    return NULL;
}

void matmul(int N, const double* __restrict__ A, const double* __restrict__ B, double* 
__restrict__ C) {
    
    pthread_t workers[NUM_THREADS - 1];
    thread_arg args[NUM_THREADS];
    int stripe = N / 4;
    int i;

    for (i = 0; i < NUM_THREADS - 1; i++) {
        args[i] = (thread_arg){ A, B, C, N, i * stripe, (i+1) * stripe };
        if (pthread_create(&workers[i], NULL, worker_func, &args[i]) != 0) {
            fprintf(stderr, "Fail to create thread!");
            exit(1);
        }
    }

    args[NUM_THREADS - 1] = (thread_arg){ A, B, C, N, (NUM_THREADS-1) * stripe, N };
    worker_func(&args[NUM_THREADS - 1]);

    for (i = 0; i < NUM_THREADS - 1; i++) {
        if (pthread_join(workers[i], NULL) != 0) {
            fprintf(stderr, "Fail to join thread!");
            exit(1);
        }
    }
}
#endif
