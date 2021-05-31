// Compile: nvcc -g -G -arch=sm_52 -std=c++11 assignment5-p3.cu -o assignment5-p3

#include <cmath>
#include <iostream>
#include <sys/time.h>

#define SIZE 4096
#define THRESHOLD (0.000001)
#define TILE 32

using std::cerr;
using std::cout;
using std::endl;

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

__host__ void ATAonCPU(double* M, double* P) {
  for (int k = 0; k < SIZE; k++) {
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++)
        P[i*SIZE + j] += M[k*SIZE + i] * M[k*SIZE + j];
    }
  }
}

__host__ void check_result(double* Test, double* Ref) {
  double maxdiff = 0, rel_diff = 0;
  int numdiffs = 0;
  
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      rel_diff = (Test[i*SIZE+j] - Ref[i*SIZE+j]);
      if (fabs(rel_diff) > THRESHOLD) {
        numdiffs++;
        if (rel_diff > maxdiff)
          maxdiff = rel_diff;
      }
    }
  }
  if (numdiffs > 0)
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << " Max Diff = " << maxdiff
         << "\n";
  else
    cout << "No differences found between base and test versions\n";
}

__global__ void ATAkernel(double* A, double* B) {
  // TODO: Fill in
  int th_x = threadIdx.x;
  int th_y = threadIdx.y;
  int b_x = blockIdx.x;
  int b_y = blockIdx.y;

  int R = b_y*TILE + th_y;
  int C = b_x*TILE + th_x;

  for(int i=0;i<SIZE;i++)
  {
	B[R*SIZE + C] += A[R*SIZE + i]*A[C*SIZE + i];
  }

  /*double sum=0;

  __shared__ double S1[TILE][TILE];
  __shared__ double S2[TILE][TILE];

  int th_x = threadIdx.x;
  int th_y = threadIdx.y;
  int b_x = blockIdx.x;
  int b_y = blockIdx.y;

  int R = b_y*TILE + th_y;
  int C = b_x*TILE + th_x;

  for(int phase=0; phase<SIZE/TILE; phase++)
  {
	S1[th_y][th_x] = A[R*SIZE + phase*TILE + th_x];
        S2[th_y][th_x] = A[C*SIZE + phase*TILE + th_y];

	__syncthreads();

	for(int t=0; t<TILE; t++)
	{
		sum += S1[th_y][t]*S2[t][th_x];
	}
	__syncthreads();
  }

  B[R*SIZE + C] = sum;
  */
}

int main() {
  cout << "Matrix Size = " << SIZE << "\n";

  //double** h_in = new double*[SIZE];
  
  double* h_in_linear = new double[SIZE*SIZE];
  
  /*for (int i = 0; i < SIZE; i++) {
    h_in[i] = new double[SIZE];
  }*/

  //double** h_cpu_out = new double*[SIZE];
  
  double* h_cpu_out_linear = new double[SIZE*SIZE];
  
  /*for (int i = 0; i < SIZE; i++) {
    h_cpu_out[i] = new double[SIZE];
  }*/

  /*double** h_dev_out = new double*[SIZE];  
  
    
  /*for (int i = 0; i < SIZE; i++) {
    h_dev_out[i] = new double[SIZE];
  }*/

  double* h_dev_out_linear = new double[SIZE*SIZE];

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
     // h_in[i][j] = i * j * 0.25;
      
        h_in_linear[i*SIZE + j] = i*j*0.25;
     
	// h_cpu_out[i][j] = 0;
      
	h_cpu_out_linear[i*SIZE + j] = 0;
     
	// h_dev_out[i][j] = 0;
      
	h_dev_out_linear[i*SIZE + j] = 0;
    }
  }

  double clkbegin = rtclock();
  ATAonCPU(h_in_linear, h_cpu_out_linear);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "A^T.A on CPU: " << ((2.0 * SIZE * SIZE * SIZE) / cpu_time)
       << " GFLOPS; Time = " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;
  double* d_in;
  double* d_out;
  float kernel_time=0;
  // TODO: Fill in
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaMalloc((void**)&d_in, SIZE*SIZE*sizeof(double));
  cudaMalloc((void**)&d_out, SIZE*SIZE*sizeof(double));

  dim3 tpb(TILE,TILE);
  dim3 bpg(SIZE/tpb.x,SIZE/tpb.y);
  cudaEventRecord(start,0);
  cudaMemcpy(d_in, h_in_linear, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_dev_out_linear, SIZE*SIZE*sizeof(double), cudaMemcpyHostToDevice);  

  ATAkernel<<<bpg,tpb>>>(d_in, d_out);
  
  cudaMemcpy(h_dev_out_linear, d_out, SIZE*SIZE*sizeof(double), cudaMemcpyDeviceToHost);
  cudaEventRecord(end,0);
  cudaDeviceSynchronize();

  cudaEventElapsedTime(&kernel_time, start, end);
  cout << "A^T.A on GPU: " << ((2.0 * SIZE * SIZE * SIZE) / (kernel_time * 1.0e-03))
       << " GFLOPS; Time = " << kernel_time << " msec" << endl;

  /*for(int i=0; i<SIZE; i++)
  {
	for(int j=0; j<SIZE; j++)
	{
	h_dev_out[i][j] = h_dev_out_linear[i*SIZE + j];
	}
	//cout<<endl;
  }*/

  /*for(int i=0; i<SIZE; i++)
  {
        for(int j=0; j<SIZE; j++)
        {
//        cout<<h_dev_out[i][j]<<" ";
        }
        cout<<endl;
  }*/
  check_result(h_cpu_out_linear, h_dev_out_linear);

  cudaFree(d_in);
  cudaFree(d_out);

  return EXIT_SUCCESS;
}
