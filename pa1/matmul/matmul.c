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
#include <string.h>
#include <pthread.h>

#define NUM_THREADS 4
#define BLOCK_SIZE 32
#define BIG_BLOCK 128
/*****************************
* Single-threaded
*****************************/
#if 1
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

void matmul_strassen_leaf(int r0, int c0, int N, const double* AA, 
                          const double* BB,  double* CC);

  void matmul_strassen(int r0, int c0, int n,  int N, 
                     const double* A, 
                     const double* B, 
                     double* C);

int isPowerOfTwo(int x){
   return x && (!(x &(x-1)));
}

int divides(int x, int y){
  return x%y == 0;
}

void print_mat(int N, const double *C){
  int k,j;
  for ( k=0; k<N;k++){
    for ( j=0; j<N;j++){
      //printf("[%i,%i] %i = %f\n",k,j,C[k*N+j],k*N+j);
      printf("%f ",k,j,C[k*N+j],k*N+j);
    }
    printf("\n");
  }
  printf("\n");
}

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

void matmul1(int N, const double* A, const double* B, double* C) {
  int i, j, k;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < N; k++)
        C[i*N + j] += A[i*N + k] * B[k*N + j];
}


void matmul_strassen_test(const double* A, const double* B,  double* C){
  C[0]+= A[0]*B[0] + A[1]*B[2];
  C[1]+= A[0]*B[1] + A[1]*B[3];
  C[2]+= A[2]*B[0] + A[3]*B[2];
  C[3]+= A[2]*B[1] + A[3]*B[3];
}

void matmul_strassen_leaf(int r0, int c0, int N, const double* AA, 
                          const double* BB,  double* C){
  // unsafe operations assuming A, B and C has size 4
  double m1,m2,m3,m4,m5,m6,m7;
  
  double A[4] = {AA[r0*N + c0],AA[r0*N +c0+1],
                 AA[(r0+1)*N+c0], AA[(r0+1)*N+c0+1]};
  double B[4] = {BB[r0*N + c0], BB[r0*N +c0+1],
                 BB[(r0+1)*N+c0], BB[(r0+1)*N+c0+1]};
  /*double GT[4] = {C[r0*N + c0], C[r0*N +c0+1],
                 C[(r0+1)*N+c0], C[(r0+1)*N+c0+1]};*
  matmul_strassen_test(&A, &B, &GT);*/

  printf("leaf index: r0=%i c0=%i N=%i [%i %i; %i %i]\n",
         r0,c0,N,
         r0*N+c0, r0*N+c0+1,(r0+1)*N+c0, (r0+1)*N + c0+1);
  m1 = (A[0]+A[3])*(B[0]+B[3]);
  m2 = (A[2]+A[3])*B[0];
  m3 = A[0]*(B[1]-B[3]);
  m4 = A[3]*(B[2]-B[0]);
  m5 = (A[0]+A[1])*B[3];
  m6 = (A[2]-A[0])*(B[0]+B[1]);
  m7 = (A[1]-A[3])*(B[2]+B[3]);
  //  C[0] = 0; C[1]=0;C[2]=0;C[3] = 0;
  printf("C: [%i %i; %i %i] = [%f %f; %f %f]\n",
         r0*N+c0, r0*N+c0+1,(r0+1)*N+c0, (r0+1)*N+c0+1,
         C[r0*N+c0], C[r0*N+c0+1], C[(r0+1)*N+c0], C[(r0+1)*N + c0+1]);

  C[r0*N + c0] += m1+m4-m5+m7;
  C[r0*N + c0 +1] += m3+m5;
  C[(r0+1)*N +c0] += m2+m4;
  C[(r0+1)*N+c0+1] += m1-m2+m3+m6; 
  printf("C: [%i %i; %i %i] = [%f %f; %f %f]\n",
         r0*N+c0, r0*N+c0+1,(r0+1)*N+c0, (r0+1)*N+c0+1,
         C[r0*N+c0], C[r0*N+c0+1], C[(r0+1)*N+c0], C[(r0+1)*N + c0+1]);
  //printf("C_gt: [%f %f; %f %f]\n",
  //       GT[0], GT[1], GT[2], GT[3]);
}

/**
 * function: matmul_strassen
 * 
 * r0: min row index of current partition
 * c0: min col index of current partition
 * n: current partition size
 * N: size of A,B and C
 */

void matmul_strassen(int r0, int c0, int n, int N , const double* A, 
                     const double* B, 
                     double* C){
  int s = N*N;
  
  //double C_copy[s];
  //memcpy(&C_copy,C,sizeof(double)*s);
  if (n==2){
    //printf("C = [%f %f; %f %f]\n",C[0],C[1],C[2],C[3]);
    matmul_strassen_leaf(r0,c0,N,A,B,C);
    //printf("C' = [%f %f; %f %f]\n",C[0],C[1],C[2],C[3]);
  }else if(isPowerOfTwo(n)){
    int new_n = n/2;
    printf("r0 = %i, c0 = %i, n = %i, N = %i , new n = %i\n",
           r0,c0,n,N,new_n);

  print_mat(N,C);   
    matmul_strassen(r0,c0,new_n,N,A,B,C);
  print_mat(N,C);      
  
  matmul_strassen(r0,c0+new_n,new_n,N,A,B,C);
  print_mat(N,C);   
    matmul_strassen(r0+new_n,c0,new_n,N,A,B,C);
print_mat(N,C);   
    matmul_strassen(r0+new_n,c0+new_n, new_n,N,A,B,C);
print_mat(N,C);   
  }
  
  
    //matmul1(N,A,B,C_copy);
  
  

  
}





void matmul(int N, const double*__restrict__ A, const double*__restrict__
B, double* __restrict__ C) {
  print_mat(N,A);

  print_mat(N,B);
  //if (N==2){
    matmul_strassen(0,0,N,N,A,B,C);
    return;
    //}



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
#if 0
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
