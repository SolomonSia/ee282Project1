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

#define DEBUG (0)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

void r_matmul(int origN, int N,int aRowStart,int aColStart,int bColStart,const double*__restrict__ A, const double*__restrict__ B, double*__restrict__ C);


void r_matmul(int origN, int N,int aRowStart,int aColStart,int bColStart,const double*__restrict__ A, const double*__restrict__ B, double*__restrict__ C){

   int start = 0;
   int midPoint = N/2;
   int iStart1,jStart1;
   int i,j,k,count;
// base case
if(N<=32){
   for(i = aRowStart;i<aRowStart+N;i++){
     for(j= bColStart;j<bColStart+N;j++){
	for(k=aColStart;k<aColStart+N;k++){
	   C[i*origN+j] += A[i*origN+k] * B[k*origN+j];
}}}
}
else{
/* test code correctness */



   for(count = 0;count<4;count++){
      if (count == 1){
	iStart1 = start;
	jStart1 = midPoint;
      }
      if (count == 2){
	iStart1 = midPoint;
	jStart1 = start;
      }
      if(count == 3){
	iStart1 = midPoint;
	jStart1 = midPoint;
      }
      if (count ==0){
	iStart1 = start;
	jStart1 = start;
      }
      r_matmul(origN,N/2,iStart1+aRowStart,start+aColStart,jStart1+bColStart,A,B,C);
      r_matmul(origN,(N+1)/2,iStart1+aRowStart,midPoint+aColStart,jStart1+bColStart,A,B,C);
   }
}
}

#if 0
void matmul(int N, const double*__restrict__ A, const double*__restrict__ B, double*__restrict__ C) {
r_matmul(N,N,0,0,0,A,B,C);
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
	 }
  }
}
}

typedef struct {
   const double * __restrict__ A;
   const double  * __restrict__ B;
   double * __restrict__ C;
   int origN;
   int curN;
   int curN2;
   int iStart1;
   int kStart1;
   int kStart2;
   int jStart1;
} thread_arg;

void * worker_func (void * __restrict__ arg) {
   thread_arg * __restrict__ targ = (thread_arg * __restrict__ )arg;

   const double * __restrict__ A = targ->A;
   const double * __restrict__ B = targ->B;
   double * __restrict__ C = targ->C;
   int origN = targ->origN;
   int curN = targ->curN;
   int curN2 = targ->curN2;
   int iStart1 = targ->iStart1;
   int kStart1 = targ->kStart1;
   int kStart2 = targ->kStart2;
   int jStart1 = targ->jStart1;

   r_matmul(origN,curN,iStart1,kStart1,jStart1,A,B,C);
   r_matmul(origN,curN2,iStart1,kStart2,jStart1,A,B,C);

   return NULL;
}

void matmul(int N, const double* __restrict__ A, const double* __restrict__ B,
		double* __restrict__ C) {
   pthread_t workers[NUM_THREADS - 1];
   thread_arg args[NUM_THREADS];
   int i,j,k;
   int origN = N;

if(N<=32){
   for(i = 0;i<N;i++){
     for(j= 0;j<N;j++){
	for(k=0;k<N;k++){
	   C[i*origN+j] += A[i*origN+k] * B[k*origN+j];
}}}
}
else{
   int start = 0;
   int midPoint = N/2;

/* test code correctness */
      args[0] = (thread_arg){A,B,C,origN,N/2,(N+1)/2,start,0,midPoint,N/2};
      pthread_create(&workers[0],NULL,worker_func,&args[0]);
      args[1] = (thread_arg){A,B,C,origN,N/2,(N+1)/2,N/2,0,midPoint,start};
      pthread_create(&workers[1],NULL,worker_func,&args[1]);
      args[2] = (thread_arg){A,B,C,origN,N/2,(N+1)/2,0,0,midPoint,0};
      pthread_create(&workers[2],NULL,worker_func,&args[2]);
   
   args[3] = (thread_arg){ A, B, C,origN, N/2,(N+1)/2,midPoint,0,midPoint,midPoint};
   worker_func(&args[3]);

   for (i = 0; i < NUM_THREADS - 1; i++) {
       if (pthread_join(workers[i], NULL) != 0) {
           fprintf(stderr, "Fail to join thread!");
           exit(1);
       }
   }


}}
#endif
