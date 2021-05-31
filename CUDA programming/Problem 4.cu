// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p4.cu -o assignment5-p4

#include <cmath>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (1 << 12);
#define THRESHOLD (0.000001)
#define blockSize 32
#define TILE 8

using std::cerr;
using std::cout;
using std::endl;

__global__ void kernel1(uint64_t* d_A, uint64_t* d_B, uint64_t* d_C) {
  // TODO: Fill in
	uint64_t R = blockIdx.y*blockDim.y + threadIdx.y;
	uint64_t C = blockIdx.x*blockDim.x + threadIdx.x;

	uint64_t sum=0;
	for(uint64_t k=0; k<N; k++)
	{
		sum += d_A[R*N + k]*d_B[k*N + C];
	}
	d_C[R*N + C] = sum;
}

__global__ void kernel2(uint64_t* d_A, uint64_t* d_B, uint64_t* d_C) {
  // TODO: Fill in
	uint64_t sum=0;
	__shared__ uint64_t A[TILE][TILE];
	__shared__ uint64_t B[TILE][TILE];

	uint64_t R = blockIdx.y*blockDim.y + threadIdx.y;
        uint64_t C = blockIdx.x*blockDim.x + threadIdx.x;
	
	for(int ph=0; ph < N/TILE; ph++)
	{
		A[threadIdx.y][threadIdx.x] = d_A[R*N + ph*TILE + threadIdx.x];
		B[threadIdx.y][threadIdx.x] = d_B[(threadIdx.y + ph*TILE)*N + C];
		
		__syncthreads();
		
		for(int k=0; k<TILE; k++)
		{
			sum += A[threadIdx.y][k]*B[k][threadIdx.x];
		}
		__syncthreads();
	}
	d_C[R*N + C] = sum;
}

__host__ void cpumatMul(uint64_t* h_A, uint64_t* h_B, uint64_t* h_C) {
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      uint64_t sum = 0;
      for (uint64_t k = 0; k < N; k++) {
        sum += h_A[i * N + k] * h_B[k * N + j];
      }
      h_C[i * N + j] = sum;
    }
  }
}

__host__ void check_result(uint64_t* w_ref, uint64_t* w_opt) {
  bool wrong = false;
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      if (w_ref[i * N + j] != w_opt[i * N + j]) {
        wrong = true;
        goto out;
      }
    }
  }
out:
  if (wrong) {
    cout << " Diffs found!" << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
  uint64_t SIZE = N * N;

  uint64_t *h_A, *h_B, *h_cpu_C, *h_gpu1_C, *h_gpu2_C;

  h_A = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_B = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_cpu_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu1_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu2_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      h_A[i * N + j] = rand() % 64;
      h_B[i * N + j] = 2;
      h_cpu_C[i * N + j] = 0;
      h_gpu1_C[i * N + j] = 0;
      h_gpu2_C[i * N + j] = 0;
    }
  }


  double clkbegin = rtclock();
  cpumatMul(h_A, h_B, h_cpu_C);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Matmul time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  
  uint64_t *d_A, *d_B, *d_C1;
  status = cudaMalloc(&d_A, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  status = cudaMalloc(&d_B, SIZE * sizeof(uint64_t));
  status = cudaMalloc(&d_C1, SIZE * sizeof(uint64_t));

  dim3 tpb1(blockSize,blockSize);
  dim3 bpg1(N/tpb1.x,N/tpb1.y);
  cudaEventRecord(start,0);

  status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
  status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

  // TODO: Fill in
  kernel1<<<bpg1, tpb1>>>(d_A, d_B, d_C1);
  cudaMemcpy(h_gpu1_C, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaEventRecord(end,0);
  cudaDeviceSynchronize();

  check_result(h_cpu_C, h_gpu1_C);
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

  uint64_t* d_C2;
  status = cudaMalloc(&d_C2, SIZE * sizeof(uint64_t));

  dim3 tpb2(TILE,TILE);
  dim3 bpg2(N/tpb2.x,N/tpb2.y);
  cudaEventRecord(start,0);

  // TODO: Fill in
  kernel2<<<bpg2,tpb2>>>(d_A, d_B, d_C2);
  cudaMemcpy(h_gpu2_C, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaEventRecord(end,0);

  check_result(h_cpu_C, h_gpu2_C);
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C1);
  cudaFree(d_C2);

  free(h_A);
  free(h_B);
  free(h_cpu_C);
  free(h_gpu1_C);
  free(h_gpu2_C);

  return EXIT_SUCCESS;
}
