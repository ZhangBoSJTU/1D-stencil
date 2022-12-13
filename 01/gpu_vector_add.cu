/*
 * Simple CPU program to add two long vectors
 *
 *
 */

#include "common/timer_nv.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

__global__ void vector_add_gpu(const int n, const float *a, const float *b,
                               float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n)
    c[tid] = a[tid] + b[tid];
}

void vector_add_cpu(const int n, const float *a, const float *b, float *c) {
  for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    perror("Command-line usage: executableName <vector size>");
    exit(1);
  }

  const int n = atof(argv[1]);

  float *x, *y, *z;

  cudaMallocManaged(&x, n * sizeof(*x));
  cudaMallocManaged(&y, n * sizeof(*y));
  cudaMallocManaged(&z, n * sizeof(*z));

  // inital data
  for (int i = 0; i < n; i++) {
    x[i] = 3.5;
    y[i] = 1.5;
  }

  for (int i = 0; i < n; i++) {
    z[i] = 0.0;
  }

  StartTimer();
  vector_add_cpu(n, x, y, z);
  // elapsed time is in seconds
  double cpu_elapsedTime = GetTimer();
  printf("vector_add on the CPU. z[100] = %4.2f\n", z[100]);
  printf("elapsed wall time (CPU) = %5.4f ms\n", cpu_elapsedTime * 1000.);

  // WARNING!!! use events only to time the device
  cudaEvent_t timeStart, timeStop;
  cudaEventCreate(&timeStart);
  cudaEventCreate(&timeStop);

  // make sure it is of type float, precision is milliseconds (ms) !!!
  float gpu_elapsedTime;

  int blockSize = 256;
  // round up if n is not a multiple of blocksize
  int nBlocks = (n + blockSize - 1) / blockSize;

  // don't worry for the 2nd argument zero, it is about cuda streams
  cudaEventRecord(timeStart, NULL);
  vector_add_gpu<<<nBlocks, blockSize>>>(n, x, y, z);
  // need this if you use cudaMallocManaged because we are not calling
  // cudaMemCpy anymore.
  cudaDeviceSynchronize();
  cudaEventRecord(timeStop, 0);
  cudaEventSynchronize(timeStop);

  printf("vector_add on the GPU. z[100] = %4.2f\n", z[100]);
  // WARNING!!! do not simply print (timeStop-timeStart)!!

  cudaEventElapsedTime(&gpu_elapsedTime, timeStart, timeStop);
  printf("elapsed wall time (GPU) = %5.4f ms\n", gpu_elapsedTime);

  // not need anymore
  cudaEventDestroy(timeStart);
  cudaEventDestroy(timeStop);

  cudaFree(x);
  cudaFree(y);
  cudaFree(z);

  return EXIT_SUCCESS;
}
