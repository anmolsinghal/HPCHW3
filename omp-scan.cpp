#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow


void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* out, const long* in, long n) 
{   
    if(n==0)
    return;
    long* suma;    
    #pragma omp parallel
    {
        int ithread = omp_get_thread_num();  
        int nthreads = omp_get_num_threads();
        #pragma omp single
        { 
          printf("Num of threads %d\n", nthreads);
          suma = (long*)malloc(sizeof *suma * (nthreads+1)), suma[0] = 0;
        }
        
        long s = 0;
        #pragma omp for schedule(static) nowait
        for (long i=1; i<n; i++) s += in[i-1], out[i] = s;
        
        suma[ithread+1] = s;

        #pragma omp barrier

        long offset = 0;
        for(long i=0; i<(ithread+1); i++) offset += suma[i];

        #pragma omp for schedule(static)
        for (long i=1; i<n; i++) out[i] += offset;
    }
   
}

int main() {
  omp_set_num_threads(4);
  long N = 100000000;

  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);
 
  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
