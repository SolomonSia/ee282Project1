/********************************************************************
* EE282 Programming Assignment 1:
* Optimization of Matrix Multiplication
*
* Updated by: mgao12    04/14/2014
********************************************************************/
#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef BLAS
#include "cblas.h"
#endif
#include "utils.h"

#define MAX_ERROR           (2.0)
#define CACHE_LINE_SIZE     (64)

// matmul() declared here and implemented in matmul.c
void matmul(int N, const double *A, const double *B, double *C);


// Magic operations for Zsim simulation
inline void ROIBegin() {
    __asm__ volatile ("movl %0, %%ecx\n\t"  \
            "xchg %%rcx, %%rcx\n\t" \
            : /* no output */   \
            : "ic" (1025 /* ROI_BEGIN */)   \
            : "%ecx"    \
            );
}

inline void ROIEnd() {
    __asm__ volatile ("movl %0, %%ecx\n\t"  \
            "xchg %%rcx, %%rcx\n\t" \
            : /* no output */   \
            : "ic" (1026 /* ROI_END */)   \
            : "%ecx"    \
            );
}

// This function is for checking matmul() for correctness.
void naive_matmul(int n, const double *A, const double *B, double *C) {
    int i, j, k;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
}

void check_correct (int n) {
    double *A, *B, *C;
    double *cA, *cB, *cC;
    int nbytes = sizeof(double) * SQR(n);

    // Allocate
    A  = (double*) malloc(nbytes);
    B  = (double*) malloc(nbytes);
    C  = (double*) malloc(nbytes);
    cA = (double*) malloc(nbytes);
    cB = (double*) malloc(nbytes);
    cC = (double*) malloc(nbytes);

    if (A  == NULL || B  == NULL || C  == NULL ||
            cA == NULL || cB == NULL || cC == NULL) {
        fprintf(stderr, "check_correct(): malloc() failed\n");
        exit(1);
    }

    printf("Checking for correctness: %d x %d ...\n", n, n);

    // Initialize
    mat_init(A, n, n);
    mat_init(B, n, n);
    mat_init(C, n, n);

    memcpy((void *)cA, (void *)A, nbytes);
    memcpy((void *)cB, (void *)B, nbytes);
    memcpy((void *)cC, (void *)C, nbytes);

    // Calculate
    printf("Using your code ...\n");
    matmul(n, A, B, C);

#ifndef BLAS
    printf("Using naive matmul ...\n");
    naive_matmul(n, cA, cB, cC);
#else
    printf("Using BLAS libs ...\n");
    // BLAS API
    // C <-- alpha * op(A) * op(B) + beta * C
    // void cblas_<s,d,c,z>gemm (
    //      const enum CBLAS_ORDER Order,
    //      const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
    //      const int M, const int N, const int K,
    //      const SCALAR alpha,
    //      const TYPE *A, const int lda,
    //      const TYPE *B, const int ldb,
    //      const SCALAR beta,
    //      TYPE *C, const int ldc )
    cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,
            1.0, cA, n, cB, n, 1.0, cC, n);
#endif

    // Compare
    double err;
    if (memcmp((void *)A, (void *)cA, nbytes) != 0 ||
            memcmp((void *)B, (void *)cB, nbytes) != 0) {
        printf("FAILED\nSource matrices have changed.\n");
    } else if ((err = error(C, cC, n, n)) > MAX_ERROR) {
        printf("FAILED\nCalculated error %f > %f\n", err, MAX_ERROR);
    } else {
        printf("PASSED\n");
    }

    // Free
    free(A); free(B); free(C);
    free(cA); free(cB); free(cC);
}

void measure_performance (int n) {
    double *A,  *B,  *C;
    double *oA, *oB, *oC;
    int i;

    printf("Measure the performance for size %d x %d ...\n", n, n);

    // Allocate
    int ret = 0;
    ret |= posix_memalign((void **)&A, CACHE_LINE_SIZE, SQR(n) * sizeof(double));
    ret |= posix_memalign((void **)&B, CACHE_LINE_SIZE, SQR(n) * sizeof(double));
    ret |= posix_memalign((void **)&C, CACHE_LINE_SIZE, SQR(n) * sizeof(double));
    if (ret) {
        fprintf(stderr, "measure_performance: posix_memalign() failed\n");
        exit(1);
    }
    oA = A; oB = B; oC = C;

    // Fill matrices with random data.
    mat_init(A, n, n);
    mat_init(B, n, n);
    mat_init(C, n, n);

    // Flush the cache
    int num = 4 * 1024 * 1024;
    int * garbage = (int *)malloc(num * sizeof(int)); // 4M * 4B = 16MB
    garbage[0] = 0;
    for (i = 1; i < num; i++) {
        garbage[i] = garbage[i - 1] + i;
    }
    free(garbage);

    // Decide how many times to repeat the measurement for average
    // For small sizes, we repeat more times, because the variance is large
    // For large sizes, we repeat fewer times, because of smaller variance and
    // the long simulation time
    int iter, max_iter;
    if (n < 128) max_iter = 100;
    else if (n < 512) max_iter = 10;
    else max_iter = 1;

    for (iter = 0; iter < max_iter; iter++) {
        ROIBegin();
        matmul(n, A, B, C);
        ROIEnd();
    }

    printf("Measurement done.\n");

    // Free
    free(oA); free(oB); free(oC);
}

void usage() {
    char help_text[] =
        "EE282 Programming Assignment 1 -- Matrix Multiplication\n\
        \n\
        Usage: matmul -s [size] <-c> <-h>\n\
        -s    Size of the matrix: N\n\
        -c    Check matmul() for correctness.\n\
        -h    Display this help text.\n\
        \n";
    fwrite(help_text, sizeof(char), strlen(help_text), stdout);
    exit(1);
}

int main(int argc, char ** argv) {
    int size = -1;
    int check_mode = 0;
    int c;

    // Initialize random number seed for mat_init
    rseed();

    while (1) {
        if ((c = getopt(argc, argv, "s:ch")) == -1)
            break;

        switch (c) {
            case 's':
                size = (int)strtol(optarg, (char **)NULL, 10);
                break;
            case 'c':
                check_mode = 1;
                break;
            case 'h':
            default:
                usage();
                break;
        }
    }

    // Must give the size of the matrix!
    if (size == -1) {
        usage();
    }

    if (check_mode) {
        check_correct(size);
    } else {
        measure_performance(size);
    }

    return 0;
}

