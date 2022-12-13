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
  // host copies of a, b, c
  float *a, *b, *c;
  // device copies of a, b, c
  float *d_a, *d_b, *d_c;
  int size = n * sizeof(float);

  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // Setup input values
  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(size);

  // inital data
  for (int i = 0; i < n; i++) {
    a[i] = 3.5;
    b[i] = 1.5;
  }

  for (int i = 0; i < n; i++) {
    c[i] = 0.0;
  }

  StartTimer();
  // Launch kernel on CPU
  vector_add_cpu(n, a, b, c);
  double cpu_elapsedTime = GetTimer();
  printf("vector_add on the CPU. z[100] = %4.2f\n", c[100]);
  printf("elapsed wall time (CPU) = %5.4f ms\n", cpu_elapsedTime * 1000.);
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
  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);


  vector_add_gpu<<<nBlocks, blockSize>>>(n, d_a, d_b, d_c);
    // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  cudaEventRecord(timeStop, 0);
  cudaEventSynchronize(timeStop);

  printf("vector_add on the GPU. z[100] = %4.2f\n", c[100]);
  // WARNING!!! do not simply print (timeStop-timeStart)!!

  cudaEventElapsedTime(&gpu_elapsedTime, timeStart, timeStop);
  printf("elapsed wall time (GPU) = %5.4f ms\n", gpu_elapsedTime);

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(a);
  free(b);
  free(c);
  return 0;
}