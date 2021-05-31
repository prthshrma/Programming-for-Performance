// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p1.cu -o assignment5-p1

#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <iostream>
#include <new>
#include <sys/time.h>

using namespace std;

#define THRESHOLD (0.000001)

#define SIZE1 8192
#define SIZE2 8200
#define ITER 100

using std::cerr;
using std::cout;
using std::endl;

__global__ void kernel1(double* d_k1_in, double* d_k1_out) {
  // TODO: Fill in
  int i=blockIdx.y*blockDim.y + threadIdx.y; 
  int j=blockIdx.x*blockDim.x + threadIdx.x;

  if(i>0 && i<SIZE1-1 && j<SIZE1-1)
  for (int k = 0; k < ITER; k++)
  {
     d_k1_out[i*SIZE1 + j + 1] = d_k1_in[(i-1)*SIZE1 + j + 1] + d_k1_in[i*SIZE1 + j + 1] + d_k1_in[(i+1)*SIZE1 + j + 1];
  }
}

__global__ void kernel2(double* d_k2_in, double* d_k2_out) {
  // TODO: Fill in
  int i=blockIdx.y*blockDim.y + threadIdx.y;
  int j=blockIdx.x*blockDim.x + threadIdx.x;
  
  
  if(i>0 && i<SIZE2-1 && j<SIZE2-1)
  for (int k = 0; k < ITER; k++)
  {
     d_k2_out[i*SIZE2 + j + 1] = d_k2_in[(i-1)*SIZE2 + j + 1] + d_k2_in[i*SIZE2 + j + 1] + d_k2_in[(i+1)*SIZE2 + j + 1];
  }

}


__host__ void serial1(double** h_ser_in, double** h_ser_out) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE1 - 1); i++) {
      for (int j = 0; j < (SIZE1 - 1); j++) {
        h_ser_out[i][j + 1] =
            (h_ser_in[i - 1][j + 1] + h_ser_in[i][j + 1] + h_ser_in[i + 1][j + 1]);
      }
    }
  }
}

__host__ void serial2(double** h_ser_in2, double** h_ser_out2) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE2 - 1); i++) {
      for (int j = 0; j < (SIZE2 - 1); j++) {
        h_ser_out2[i][j + 1] =
            (h_ser_in2[i - 1][j + 1] + h_ser_in2[i][j + 1] + h_ser_in2[i + 1][j + 1]);
      }
    }
  }
}

__host__ void check_result(double** w_ref, double** w_opt, uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

__host__ double rtclock() { // Seconds
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
  double** h_ser_in = new double*[SIZE1];
  double** h_ser_out = new double*[SIZE1];

  double** h_k1_in = new double*[SIZE1];
  double** h_k1_out = new double*[SIZE1];
  double* h_k1_in_linear = new double[SIZE1*SIZE1];
  double* h_k1_out_linear = new double[SIZE1*SIZE1];

  for (int i = 0; i < SIZE1; i++) {
    h_ser_in[i] = new double[SIZE1];
    h_ser_out[i] = new double[SIZE1];
    h_k1_in[i] = new double[SIZE1];
    h_k1_out[i] = new double[SIZE1];
  }

  for (int i = 0; i < SIZE1; i++) {
    for (int j = 0; j < SIZE1; j++) {
      h_ser_in[i][j] = 1;
      h_ser_out[i][j] = 0;
      h_k1_in[i][j] = 1;
      h_k1_out[i][j] = 0;
    }
  }

  double** h_ser_in2 = new double*[SIZE2];
  double** h_ser_out2 = new double*[SIZE2];

  double** h_k2_in = new double*[SIZE2];
  double** h_k2_out = new double*[SIZE2];
  double* h_k2_in_linear = new double[SIZE2*SIZE2];
  double* h_k2_out_linear = new double[SIZE2*SIZE2];
  for (int i = 0; i < SIZE2; i++) {
    h_ser_in2[i] = new double[SIZE2];
    h_ser_out2[i] = new double[SIZE2];
    h_k2_in[i] = new double[SIZE2];
    h_k2_out[i] = new double[SIZE2];
  }

  for (int i = 0; i < SIZE2; i++) {
    for (int j = 0; j < SIZE2; j++) {
      h_ser_in2[i][j] = 1;
      h_ser_out2[i][j] = 0;
      h_k2_in[i][j] = 1;
      h_k2_out[i][j] = 0;
    }
  }


  
  // converting host 2d array to linear
  for (int i = 0; i < SIZE1; i++) {
    for (int j = 0; j < SIZE1; j++) {
        h_k1_in_linear[i*SIZE1 + j]=h_k1_in[i][j];
//        h_k2_in_linear[i*SIZE1 + j]=h_k2_in[i][j];
        h_k1_out_linear[i*SIZE1 + j]=h_k1_out[i][j];
//        h_k2_out_linear[i*SIZE1 + j]=h_k2_out[i][j];
    }
  }

  for(int i=0;i<SIZE2;i++)
  {
	for(int j=0;j<SIZE2;j++)
	{
		h_k2_in_linear[i*SIZE2 + j]=h_k2_in[i][j];
		h_k2_out_linear[i*SIZE2 + j]=h_k2_out[i][j];
	}
  }

  double clkbegin = rtclock();
  serial1(h_ser_in, h_ser_out);
  double clkend = rtclock();
  double time = clkend - clkbegin; // seconds
  cout << "Serial code on CPU for SIZE1: " << ((2.0 * SIZE1 * SIZE1 * ITER) / time)
       << " GFLOPS; Time = " << time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  float k1_time; // ms

  double* d_k1_in;
  double* d_k1_out;
  // TODO: Fill in
    
  cudaMalloc((void**)&d_k1_in, SIZE1*SIZE1*sizeof(double));
  cudaMalloc((void**)&d_k1_out, SIZE1*SIZE1*sizeof(double));  

  cudaEventRecord(start, 0);
  cudaMemcpy(d_k1_in, h_k1_in_linear, SIZE1*SIZE1*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k1_out, h_k1_out_linear, SIZE1*SIZE1*sizeof(double), cudaMemcpyHostToDevice);
  
  dim3 threadsPerBlock1(32,32);
  dim3 blocksPerGrid1(SIZE1/threadsPerBlock1.x,SIZE1/threadsPerBlock1.y);
  
  kernel1<<<blocksPerGrid1,threadsPerBlock1>>>(d_k1_in,d_k1_out);
  cudaDeviceSynchronize();
  cudaMemcpy(h_k1_in_linear, d_k1_in, SIZE1*SIZE1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_k1_out_linear, d_k1_out, SIZE1*SIZE1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  
  k1_time=0;
  for(int i=0;i<SIZE1;i++)  
  {
	for(int j=0;j<SIZE1;j++)
	h_k1_out[i][j]=h_k1_out_linear[i*SIZE1+j];
  }  
  cudaEventElapsedTime(&k1_time, start, end);
  check_result(h_ser_out, h_k1_out, SIZE1);
  cout << "Kernel 1 on GPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / (k1_time * 1.0e-3))<< " GFLOPS; Time = " << k1_time << " msec" << endl;



  clkbegin = rtclock();
  serial2(h_ser_in2, h_ser_out2);
  clkend = rtclock();
  time = clkend - clkbegin; // seconds
  cout << "Serial code on CPU for SIZE 2: " << ((2.0 * SIZE2 * SIZE2 * ITER) / time) << " GFLOPS; Time = " << time * 1000 << " msec" << endl;
  
  double* d_k2_in;
  double* d_k2_out;

  cudaMalloc((void**)&d_k2_in, SIZE2*SIZE2*sizeof(double));
  cudaMalloc((void**)&d_k2_out, SIZE2*SIZE2*sizeof(double));

  // TODO: Fill in
  cudaEventRecord(start, 0);
  cudaMemcpy(d_k2_in, h_k2_in_linear, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k2_out, h_k2_out_linear, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock2(32,32);
  dim3 blocksPerGrid2((SIZE2+threadsPerBlock2.x-1)/threadsPerBlock2.x,(SIZE2+threadsPerBlock2.y-1)/threadsPerBlock2.y);

  kernel2<<<blocksPerGrid2,threadsPerBlock2>>>(d_k2_in,d_k2_out);
  cudaDeviceSynchronize();
  cudaMemcpy(h_k2_in_linear, d_k2_in, SIZE2*SIZE2*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_k2_out_linear, d_k2_out, SIZE2*SIZE2*sizeof(double), cudaMemcpyDeviceToHost);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  k1_time=0;
  for(int i=0;i<SIZE2;i++)
  {
        for(int j=0;j<SIZE2;j++)
        h_k2_out[i][j]=h_k2_out_linear[i*SIZE2+j];
  }
  cudaEventElapsedTime(&k1_time, start, end);
  check_result(h_ser_out2, h_k2_out, SIZE2);
  cout << "Kernel 2 on GPU: " << ((2.0 * SIZE2 * SIZE2 * ITER) / (k1_time * 1.0e-3))
       << " GFLOPS; Time = " << k1_time << " msec" << endl;

  cudaFree(d_k1_in);
  cudaFree(d_k1_out);
  cudaFree(d_k2_in);
  cudaFree(d_k2_out);

  for (int i = 0; i < SIZE1; i++) {
    delete[] h_ser_in[i];
    delete[] h_ser_out[i];
    delete[] h_k1_in[i];
    delete[] h_k1_out[i];
  }
  delete[] h_ser_in;
  delete[] h_ser_out;
  delete[] h_k1_in;
  delete[] h_k1_out;

  for (int i = 0; i < SIZE2; i++) {
    delete[] h_k2_in[i];
    delete[] h_k2_out[i];
  }
  delete[] h_k2_in;
  delete[] h_k2_out;

  return EXIT_SUCCESS;
}
