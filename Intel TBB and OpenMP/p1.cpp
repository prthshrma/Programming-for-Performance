// Compile: g++ -o 20111039-p1.out 20111039-p1.cpp -fopenmp -pthread -ltbb -I ./tbb/include -L ./tbb/lib
// Run: ./20111039-p1.out

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <bits/stdc++.h>
#include <tbb/task.h>

using namespace tbb;

class FibConti: public task {
public:
long* const sum;
long fibChild1, fibChild2;
FibConti( long* Sum ) : sum(Sum) {}
  
  task* execute( ) {
  *sum = fibChild1 + fibChild2;
  return NULL;
  }
};


class Fib_Continuation: public task {
 public:
  const long n;
  long* const sum;
  Fib_Continuation( long Num, long* S ) : n(Num), sum(S) {}

  long ser_fib(int n) 
  {
    if (n == 0 || n == 1) {
    return (n);
    }
    return ser_fib(n - 1) + ser_fib(n - 2);
  }

  task* execute( ) 
  {
    if( n<16 ) {
    *sum = ser_fib(n);
    return NULL;
    } 
    else {
    FibConti& conti = *new( allocate_continuation( ) ) FibConti(sum);
    Fib_Continuation& fibChild1 = *new( conti.allocate_child( ) ) Fib_Continuation(n-2,&conti.fibChild1);
    Fib_Continuation& fibChild2 = *new( conti.allocate_child( ) ) Fib_Continuation(n-1,&conti.fibChild2);
  
    conti.set_ref_count(2);
    conti.spawn( fibChild2 );
    conti.spawn( fibChild1 );
    return NULL;
    }
  }
};


class Fib_Blocking: public task {
  public:
    const long n;
    long* const sum;
    Fib_Blocking( long Num, long* S ) : n(Num), sum(S) {}
    
    	long ser_fib(int n) {
  	if (n == 0 || n == 1) {
    	return (n);
  	}
  	return ser_fib(n - 1) + ser_fib(n - 2);
	}
    
    task* execute()
    {
      if(n<16)
        *sum = ser_fib(n);
      else
      {
        long sum1,sum2;
        Fib_Blocking& fibChild1 = *new( allocate_child() ) Fib_Blocking(n-1,&sum1);
        Fib_Blocking& fibChild2 = *new( allocate_child() ) Fib_Blocking(n-2,&sum2);
        
        set_ref_count(3);
        spawn(fibChild2);
        spawn_and_wait_for_all(fibChild1);
        
        *sum = sum1+sum2;
      }      
      return NULL;
    }
	
};


#define N 50

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;


// Serial Fibonacci
long long int ser_fib(int n) {
  if (n == 0 || n == 1) {
    return (n);
  }
  return ser_fib(n - 1) + ser_fib(n - 2);
}

long long int omp_fib_v1(int n) {
  // TODO: Implement OpenMP version with explicit tasks
  long long int sum1,sum2,sum;
  
	    if (n == 0 || n == 1) 
	    {
	    return n;
	    }
	    else{
	    #pragma omp task shared(sum1) 
	    sum1=omp_fib_v1(n-1);
	    
	    #pragma omp task shared(sum2)
	    sum2=omp_fib_v1(n-2);
	    
	    #pragma omp taskwait
	    sum=sum1+sum2;
	    return sum;
            }
}

long omp_fib_v2(int n) {
  // TODO: Implement an optimized OpenMP version with any valid optimization
            long long int sum1,sum2,sum;
  
	    if (n<=20) 
	    {
	    return ser_fib(n);
	    }
	    else{
	    #pragma omp task shared(sum1) 
	    sum1=omp_fib_v2(n-1);
	    
	    #pragma omp task shared(sum2)
	    sum2=omp_fib_v2(n-2);
	    
	    #pragma omp taskwait
	    sum=sum1+sum2;
	    return sum;
            }
  
}

long tbb_fib_blocking(int n) {
  // TODO: Implement Intel TBB version with blocking style
  long sum=0;
  Fib_Blocking& fibRoot = *new(task::allocate_root()) Fib_Blocking(n,&sum);
  task::spawn_root_and_wait(fibRoot);
  return sum;
}

long tbb_fib_cps(int n) {
  // TODO: Implement Intel TBB version with continuation passing style
  long sum=0;
  Fib_Continuation& fibRoot = *new(task::allocate_root()) Fib_Continuation(n,&sum);
  task::spawn_root_and_wait(fibRoot);
  return sum;
}

int main(int argc, char** argv) {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  long long int s_fib = ser_fib(N);
  HRTimer end = HR::now();
  auto sduration = duration_cast<microseconds>(end - start).count();
  cout << "Serial time: " << sduration << " microseconds" << endl;

  start = HR::now();
  long long int omp_v1;    
  omp_v1 = omp_fib_v1(N);
  end = HR::now();
  assert(s_fib == omp_v1);
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v1 time: " << duration << " microseconds" << "and speedup : "<< sduration/float(duration) << endl;

  start = HR::now();
  long omp_v2;
  #pragma omp parallel
  { 
    #pragma omp single
    omp_v2 = omp_fib_v2(N);
  }
  end = HR::now();
  assert(s_fib == omp_v2);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v2 time: " << duration << " microseconds" << "and speedup : "<< sduration/double(duration) << endl;


  start = HR::now();
  long blocking_fib = tbb_fib_blocking(N);
  end = HR::now();
  assert(s_fib == blocking_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (blocking) time: " << duration << " microseconds" << "and speedup : "<< sduration/double(duration) << endl;

  start = HR::now();
  long cps_fib = tbb_fib_cps(N);
  end = HR::now();
  assert(s_fib == cps_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (cps) time: " << duration << " microseconds" << "and speedup : "<< sduration/double(duration) << endl;

  return EXIT_SUCCESS;
}
