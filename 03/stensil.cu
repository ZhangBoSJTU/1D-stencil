#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 1000000  // job size = 1K,10K,100K,1M,10M
#define r 128      // radius = 2,4,8,16
#define r 128 // radius = 2,4,8,16
#define BLOCK_SIZE 1024    // fixed number of threads per block

// CUDA API error checking macro
static void handleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define cudaCheck(err) (handleError(err, __FILE__, __LINE__))

__global__ void stencil_1d(int *in, int *out) {

  // index of a thread across all threads + r
  int gindex = threadIdx.x + (blockIdx.x * blockDim.x) + r;

  int result = 0;
  for (int offset = -r; offset <= r; offset++)
    result += in[gindex + offset];

  // Store the result
  out[gindex - r] = result;
}

__global__ void stencil_1d_sharedMem(int *in, int *out) {
  __shared__ int temp[BLOCK_SIZE + 2 * r];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x + r;
  temp[lindex] = in[gindex]; // storing in shared memory

  if (threadIdx.x < r) {
    temp[lindex - r] = in[gindex - r];
    temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
  }
  __syncthreads();
  int result = 0;
  for (int offset = -r; offset <= r; offset++) {
    result += temp[lindex + offset];
  }
  // Store the result
  out[gindex] = result;
}

void _1d_stensil_cpu() {

  int i;
  int array[n + 4];
  int add[n];

  for (i = 0; i < n + 4; i++) {
    array[i] = rand() % n + 4;
  }
  int offset;
  int j;
  int k;
  int index = 0;
  for (j = 0; j < n; j++) {
    add[j] = 0;
  }
  clock_t begin = clock();
  for (k = r; k < n + r; k++) {

    for (offset = -r; offset <= r; offset++) {
      add[k - r] = add[k - r] + array[index + offset + r];
    }
    index++;
  }
  clock_t end = clock();
  double execution_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("CPU Execution Time = %f\n", execution_time);

}

void _1d_stensil_gpu_glb()

{
  unsigned int i;

  cudaEvent_t start, stop; // time start and stop
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // CPU array copies
  int h_in[n + 2 * r], h_out[n];

  // GPU array copies
  int *d_in, *d_out;

  for (i = 0; i < (n + 2 * r); ++i)
    h_in[i] = 1;

  // Allocate device memory
  cudaCheck(cudaMalloc(&d_in, (n + 2 * r) * sizeof(int)));
  cudaCheck(cudaMalloc(&d_out, n * sizeof(int)));

  // copy fro CPU to GPU memory
  cudaCheck(cudaMemcpy(d_in, h_in, (n + 2 * r) * sizeof(int),
                       cudaMemcpyHostToDevice));

  cudaEventRecord(start, 0);
  // Call stencil kernel
  stencil_1d<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
      d_in, d_out);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("GPU Execution Time = %f (GlobalMem)\n", time);

  // Copy results from device memory to host
  cudaCheck(cudaMemcpy(h_out, d_out, n * sizeof(int),
                       cudaMemcpyDeviceToHost));

  // Free out memory
  cudaFree(d_in);
  cudaFree(d_out);

}

void _1d_stensil_gpu_shared() {
  unsigned int i;
  // CPU array copies
  int h_in[n + 2 * r], h_out[n];
  // GPU array copies
  int *d_in, *d_out;

  cudaEvent_t start, stop; // time start and stop
  float time;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (i = 0; i < (n + 2 * r); ++i)
    h_in[i] = 1;

  // Allocate device memory
  cudaCheck(cudaMalloc(&d_in, (n + 2 * r) * sizeof(int)));
  cudaCheck(cudaMalloc(&d_out, n * sizeof(int)));

  // copy fro CPU to GPU memory
  cudaCheck(cudaMemcpy(d_in, h_in, (n + 2 * r) * sizeof(int),
                       cudaMemcpyHostToDevice));
  cudaEventRecord(start, 0);

  // Call stencil kernel
  stencil_1d_sharedMem<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                         BLOCK_SIZE>>>(d_in, d_out);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("GPU Execution Time = %f (SharedMem) \n", time);

  // copy from device to host
  cudaCheck(cudaMemcpy(h_out, d_out, n * sizeof(int),
                       cudaMemcpyDeviceToHost));

  // Free out memory
  cudaFree(d_in);
  cudaFree(d_out);

}

int main() {
    _1d_stensil_cpu();
    // _1d_stensil_gpu_glb();
    // _1d_stensil_gpu_shared();
    return 0;
}