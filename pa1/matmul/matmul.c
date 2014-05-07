
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
//#include <string.h>
#include <pthread.h>

#define NUM_THREADS 4
#define BLOCK_SIZE 32
#define BIG_BLOCK 128
/*****************************
* Single-threaded
*****************************/
#if 1
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define AT(A,N,i,j) (A+(i*N+j))
#define AT_REF(A,N,i,j) (*AT(A,N,i,j))


void matmul_strassen_leaf(int r0, int c0, int N, const double* AA, 
                          const double* BB,  double* CC);

void matmul_strassen(int n,  int N, 
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
      printf("%f ",C[k*N+j]);
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

void matmul1(int N,//int r0,int c0, int c1, 
             const double* A, const double* B, double* C) {
  int i, j, k;

  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      double val = 0;
      for (k = 0; k < N; k++){
        val += AT_REF(A, N, i, k) * AT_REF(B,N,k, j);
      }
      AT_REF(C,N,i,j)=val;
    }
  }
}
void matmul2(int N,//int r0,int c0, int c1, 
             const double* A, const double* B, double* C) {
  int i, j, k;

  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
      double val = 0;
      for (k = 0; k < N; k++){
        val += AT_REF(A, N, i, k) * AT_REF(B,N,k, j);
      }
      AT_REF(C,N,i,j) +=val;
    }
  }
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
         r0*N+c0, r0*N+c0+1, (r0+1)*N+c0, (r0+1)*N+c0+1,
         C[r0*N+c0], C[r0*N+c0+1], C[(r0+1)*N+c0], C[(r0+1)*N + c0+1]);
  //printf("C_gt: [%f %f; %f %f]\n",
  //       GT[0], GT[1], GT[2], GT[3]);
}
void mat_add(int n, int N,  double *A1, double *A2, double
  *sum){
  const double *array_end = A1 +n*n;
  for (;A1 < array_end;A1+=n,A2+=n,sum+=n){
    const double *row_end = A1 + n;
    for (; A1 < row_end;A1++,A2++, sum++){
      *sum = *A1 + *A2;
    }
  }
}
void mat_sub(int n, int N,  double *A1, double *A2, double
  *diff){
  const double* array_end = A1+n*n;
  for (;A1 < array_end;A1+=n,A2+=n,diff+=n){
    const double* row_end = A1 + n;
    for (; A1 < row_end;A1++,A2++,diff++){
      *diff = *A1 - *A2;
    }
  }
}

void  strassen_M1( int n, int N, double *A1,double *A2, double *B1, 
                   double* B2, double *M){
  double *sum1 = (double*)malloc(sizeof(double)*n*n);
  double *sum2 = (double*)malloc(sizeof(double)*n*n);
  mat_add(n, N, A1,A2,sum1);
  mat_add(n, N, B1,B2,sum2);
  matmul_strassen(n, N,sum1, sum2, M);
  free(sum1);
  free(sum2);
}

void  strassen_M2( int n, int N, double *A1,double *A2, 
                   double *B, double *M){
  double *sum = (double*)malloc(sizeof(double)*n*n);
  mat_add(n, N, A1,A2,sum);
  matmul_strassen(n, N,sum, B, M);
  free(sum);

}
void  strassen_M3( int n, int N, double *A,double *B1, 
                   double *B2, double *M){
  double *diff = (double*)malloc(sizeof(double)*n*n);
  mat_sub(n, N, B1, B2, diff);
  matmul_strassen(n, N, A, diff,M);
  
  free(diff);
}
void  strassen_M6( int n, int N, double *A1, double *A2,
                   double *B1, double *B2, double *M){
  double *diff = (double*)malloc(sizeof(double)*n*n);
  
  double *sum = (double*)malloc(sizeof(double)*n*n);
  mat_sub(n, N, A1, A2, diff);
  mat_add(n, N, B1, B2, sum);
  matmul_strassen(n, N,diff,sum,M);
  free(diff);free(sum);
}
// C12 = M3  +M5
void strassen_C12(int n, int N, double *M1, double *M2, double *C){
  mat_add(n,N,C,M1,C); //C = C+M1;
  mat_add(n,N,C,M2,C); //C = C+M2;
}
    // C22 = M1+M3+M6-M2
void strassen_C11(int n, int N, double *M1, double *M2, double *M3,
                  double *M6,double *C){


  mat_add(n,N,C,M1,C); //C = C+M1
  mat_sub(n,N,C,M2,C); //c = c-m2
  mat_add(n,N,C,M3,C); //c = c+M3
  mat_add(n,N,C,M6,C); //c = c+M6

} 
/**
 * function: matmul_strassen
 * 
 * r0: min row index of current partition
 * c0: min col index of current partition
 * n: current partition size
 * N: size of A,B and C
 */

void matmul_strassen( int n, int N , const double* A, 
                     const double* B, double* C){
  int s = N*N;
  
  //double C_copy[s];
  //memcpy(&C_copy,C,sizeof(double)*s);
  if (n <= 4){
    matmul1(N,A,B,C);
    
    //matmul_strassen_leaf(r0,c0,N,A,B,C);
    //printf("C' = [%f %f; %f %f]\n",C[0],C[1],C[2],C[3]);
  }else if(isPowerOfTwo(n)){
    int new_n  = n >> 1;

    double *A11 = AT(A,N,0,0);
    double *A12 = AT(A,N,0,new_n);
    double *A21 = AT(A,N,new_n,0);
    double *A22 = AT(A,N,new_n,new_n);

    double *B11 = AT(B,N,0,0);
    double *B12 = AT(B,N,0,new_n);
    double *B21 = AT(B,N,new_n,0);
    double *B22 = AT(B,N,new_n,new_n);

    double *C11 = AT(C,N,0,0);
    double *C12 = AT(C,N,0,new_n);
    double *C21 = AT(C,N,new_n,0);
    double *C22 = AT(C,N,new_n,new_n);
    
    double *M1 = (double*) malloc(sizeof(double)*new_n*new_n);
    double *M3 = (double*) malloc(sizeof(double)*new_n*new_n);
    double *M4 = (double*) malloc(sizeof(double)*new_n*new_n);

    double *M7 = (double*) malloc(sizeof(double)*new_n*new_n);
    double *M5 = (double*) malloc(sizeof(double)*new_n*new_n);
    double *M2 = (double*) malloc(sizeof(double)*new_n*new_n);
    double *M6 = (double*) malloc(sizeof(double)*new_n*new_n); 

    
    strassen_M1( new_n, N,A11, A22, B11, B22, M1);
    strassen_M2(new_n,N,A21,A22,B11,M2);
    strassen_M3(new_n,N,A11,B12,B22,M3);
    strassen_M3(new_n,N,A22,B21,B11,M4);
    strassen_M2(new_n,N,A11,A12,B22,M5);
    strassen_M6(new_n,N,A21,A11,B11,B12,M6);
    strassen_M6(new_n,N,A12,A22,B21,B22,M7);

    // C11 = M1+M4+M7-M5
    strassen_C11(new_n,N,M1,M4,M5,M7,C11);
    // C12 = M3+M5
    strassen_C12(new_n,N,M3,M5,C12);
    // C21 = M2 + M4
    strassen_C12(new_n,N,M2,M4,C21);
    // C22 = M1+M3+M6-M2
    strassen_C11(new_n,N,M1,M3,M2,M6,C22);

    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);
    free(M7);


    //    printf("r0 = %i, c0 = %i, n = %i, N = %i , new n = %i\n",
    //       r0,c0,n,N,new_n);
    /*int M_size = new_n*new_n;
    double M[M_size];
    double T[M_size];
    // A11+A22
    matadd(r0, c0, r0+new_n, c0+new_n, N,A,A,&M);
    matadd(r0, c0, r0+new_n, c0+new_n, N,B,B,&T);
    matmul_strassen(r0,c0,new_n
    // compute M1
    

    matadd(r0,c0,new_n,N,&M,C);
  
  matmul_strassen(r0,c0+new_n,new_n,N,A,B,C);
  print_mat(N,C);   
    matmul_strassen(r0+new_n,c0,new_n,N,A,B,C);
print_mat(N,C);   
    matmul_strassen(r0+new_n,c0+new_n, new_n,N,A,B,C);
print_mat(N,C);   
    */
  }
  
  
    //matmul1(N,A,B,C_copy);
  
  

  
}





void matmul(int N, const double*__restrict__ A, const double*__restrict__
B, double* __restrict__ C) {
  print_mat(N,A);

  print_mat(N,B);
  print_mat(N,C);
  //if (N==2){
  if (N== 2){
    matmul2(N,A,B,C);

  }else{
    matmul_strassen(N,N,A,B,C);
  }
  print_mat(N,C);
    //return;
}
    //}

/* matrix transpose */
//   double Bt[N][N];
//   double *__restrict__ Btt = (double *)Bt;
//   int i, j, k, i0, j0, k0,i1,j1,k1;
 /*  for(i=0;i<N;i+=BLOCK_SIZE){
    for(j=0;j<N;j+=BLOCK_SIZE){
     for(k=i;k<i+BLOCK_SIZE;++k){
      for(l=j;l<j+BLOCK_SIZE;++l){
        Btt[k+l*N] = B[l+k*N];
}}}}
*/
    /*
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
    */

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
