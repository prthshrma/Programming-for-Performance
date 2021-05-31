// g++ -o 20111039-p3.out 20111039-p3.cpp -fopenmp -pthread -ltbb -I ./tbb/include -L ./tbb/lib
// Run: ./20111039-p3.out

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

using namespace tbb;
using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

const int NUM_INTERVALS = std::numeric_limits<int>::max();


class CalculatePi
{
  public:
  double sum;
  double dx = 1.0 / NUM_INTERVALS;
  
  CalculatePi (CalculatePi& P, split) : sum(0.0) {}
  
  CalculatePi () : sum(0.0) {}
   
  void operator() (const blocked_range<int>& r) {
    for(int i=r.begin();i!=r.end();i++)
    {
      double x = (i + 0.5) * dx;
      double h = std::sqrt(1 - x * x);
      sum += h * dx;
    }
  }
  
  void join(const CalculatePi& P)
  {
    sum+=P.sum;
  }
   
};

double serial_pi() {
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;
  for (int i = 0; i < NUM_INTERVALS; ++i) {
    double x = (i + 0.5) * dx;
    double h = std::sqrt(1 - x * x);
    sum += h * dx;
  }
  double pi = 4 * sum;
  return pi;
}


double omp_pi() {
  // TODO: Implement OpenMP version with minimal false sharing
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;
  #pragma omp parallel reduction(+ : sum)
  {
  	#pragma omp for 
	  for (int i = 0; i < NUM_INTERVALS; ++i) 
	  {
	    double x = (i + 0.5) * dx;
	    double h = std::sqrt(1 - x * x);
	    sum += h * dx;
	  }
	  
  }
  double pi = 4 * sum;
  return pi;
}

double tbb_pi() {
  // TODO: Implement TBB version with parallel algorithms
  CalculatePi P;
  parallel_reduce(blocked_range<int>(0,NUM_INTERVALS), P);
  double pi = 4 * P.sum;
  return pi;
}

int main() {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  double ser_pi = serial_pi();
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial pi: " << ser_pi << " Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  double o_pi = omp_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (OMP): " << o_pi << " Parallel time: " << duration << " microseconds" << endl;

  start = HR::now();
  double t_pi = tbb_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (TBB): " << t_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  return EXIT_SUCCESS;
}

// Local Variables:
// compile-command: "g++ -std=c++11 pi.cpp -o pi; ./pi"
// End:
