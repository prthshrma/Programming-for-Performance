// Compile: g++ -o 20111039-p4.out 20111039-p4.cpp -fopenmp -pthread -ltbb -I ./tbb/include -L ./tbb/lib
// Run: 20111039-p4.out

#include <cassert>
#include <chrono>
#include <iostream>
#include <tbb/tbb.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

using namespace tbb;

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 26)


class FindMax
{
  const uint32_t *const A;
  public:
  uint32_t maxValue;
  uint32_t maxIndex;
  
  FindMax (FindMax& Max, split) : A(Max.A), maxValue(0), maxIndex(-1) {}
  
  FindMax (const uint32_t a[]) : A(a), maxValue(0), maxIndex(-1) {}
   
  void operator() (const blocked_range<uint32_t>& r) {
    const uint32_t* a=A;
    for(uint32_t i=r.begin();i!=r.end();i++)
    {
      uint32_t v=a[i];
      if(v>maxValue)
      {
        maxValue=v;
        maxIndex=i;
      }
    }
  }
  
  void join(const FindMax& Max)
  {
    if(Max.maxValue > maxValue)
    {
      maxValue=Max.maxValue;
      maxIndex=Max.maxIndex;
    }
    else if(Max.maxValue == maxValue && Max.maxIndex < maxIndex)
    {
      maxIndex=Max.maxIndex;
    }
  }
  
  
};

uint32_t serial_find_max(const uint32_t* a) {
  uint32_t value_of_max = 0;
  uint32_t index_of_max = -1;
  for (uint32_t i = 0; i < N; i++) {
    uint32_t value = a[i];
    if (value > value_of_max) {
      value_of_max = value;
      index_of_max = i;
    }
  }
  return index_of_max;
}

uint32_t tbb_find_max(const uint32_t* a) {
  // TODO: Implement a parallel max function with Intel TBB
  FindMax Imax(a);
  parallel_reduce(blocked_range<uint32_t>(0,N), Imax);
  return Imax.maxIndex;
}

int main() {
  uint32_t* a = new uint32_t[N];
  for (uint32_t i = 0; i < N; i++) {
    a[i] = i;
  }

  HRTimer start = HR::now();
  uint64_t s_max_idx = serial_find_max(a);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Sequential max index: " << s_max_idx << " in " << duration << " us" << endl;

  start = HR::now();
  uint64_t tbb_max_idx = tbb_find_max(a);
  end = HR::now();
  //cout<<"Max : "<<tbb_max_idx<<endl;
  assert(s_max_idx == tbb_max_idx);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel (TBB) max index in " << duration << " us" << endl;

  return EXIT_SUCCESS;
}
