# Exercises for Parallel and High Performance Computing

So there are exercises for this course on the following website : https://c4science.ch/diffusion/10104/.

**Week 4**

In the *debugging* folder, there is a Makefile as follows :

```
#OPTIM+=-O3 -march=native
DEBUG+=-g -O1
CXX=g++
CC=g++
LD=${CXX}
CXXFLAGS+=$(OPTIM) $(DEBUG) -Wall -Wextra -std=c++11
LDFLAGS+=$(OPTIM) $(DEBUG) $(CXXFLAGS) -lm

// because we have read.cc and write.cc file in the same folder 
all: read write

pi: read.o write.o

clean:
	rm -f read write *.o *~
```

Then we have the following *read.cc* file :

```
// the libraries are imported within <> 
#include <vector>
#include <iostream>

// in c++ the main() method returns an integer and doesn't take in any arguments
int main() {
  // The constexpr specifier declares that it is possible to evaluate the value 
  // of the function or variable at compile time.
  constexpr size_t N = 1000;
  // you specify a vector using the two dots after std, and the type inside the vector is of type double 
  std::vector<double> data(N);
  
  // size_t is the unsigned integer type of the result of the sizeof operator
  for(size_t i = 0; i < N; ++i) {
     data[i] = i;
  }

  double sum = 0.;
  for(size_t i = 0; i <= N; ++i) {
    sum += data[i];
  }
  
  // The cout object is used along with the insertion operator (<<) in order
  // to display a stream of characters. Maybe here each time you have a new << this
  // creates a new line ?
  // for the end of a line you need to write std::endl
  std::cout << (N * (N-1) / 2.) << " == " << sum << std::endl;
}
```

Now in the *pi* folder we have the `pi_reduction.cc` file which we can write as follows :

```
#include <chrono>
#include <cmath>
#include <cstdio>

#if defined(_OPENMP)
#include <omp.h>
#endif

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

inline int digit(double x, int n) {
  return std::trunc(x * std::pow(10., n)) -
         std::trunc(x * std::pow(10., n - 1)) * 10.;
}

inline double f(double a) { return (4. / (1. + a * a)); }

const int n = 10000000;

int main(int /* argc */, char ** /* argv */) {
  int i;
  double dx, x, sum, pi;

#if defined(_OPENMP)
  int num_threads = omp_get_max_threads();
#endif

#if defined(_OPENMP)
  auto omp_t1 = omp_get_wtime();
#endif
  auto t1 = clk::now();

  sum = 0.;
#pragma omp parallel shared(sum) private(x)
  {
    /* calculate pi = integral [0..1] 4 / (1 + x**2) dx */
    dx = 1. / n;
#pragma omp for reduction(+:sum)
    for (i = 1; i <= n; i++) {
      x = (1. * i - 0.5) * dx;
      sum = sum + f(x);
    }
  }
  pi = dx * sum;

#if defined(_OPENMP)
  auto omp_elapsed = omp_get_wtime() - omp_t1;
#endif
  second elapsed = clk::now() - t1;

  std::printf("computed pi                     = %.16g\n", pi);
#if defined(_OPENMP)
  std::printf("wall clock time (omp_get_wtime) = %.4gs on %d threads\n",
              omp_elapsed, num_threads);
#endif
  std::printf("wall clock time (chrono)        = %.4gs\n", elapsed.count());

  for (int d = 1; d <= 15; ++d) {
    std::printf("%d", digit(pi, d));
  }

  return 0;
}
```

Then we have the following `pi_for_wrong.cc` file as follows :

```
/*
  This exercise is taken from the class Parallel Programming Workshop (MPI,
  OpenMP and Advanced Topics) at HLRS given by Rolf Rabenseifner
 */

#include <chrono>
#include <cstdio>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

inline int digit(double x, int n) {
  return std::trunc(x * std::pow(10., n)) - std::trunc(x * std::pow(10., n - 1)) *10.;
}

inline double f(double a) { return (4. / (1. + a * a)); }

const int n = 10000000;

int main(int /* argc */ , char ** /* argv */) {
  int i;
  double dx, x, sum, pi;
  int nthreads;

#ifdef _OPENMP
  nthreads = omp_get_max_threads();
  auto omp_t1 = omp_get_wtime();
#endif
  auto t1 = clk::now();

  /* calculate pi = integral [0..1] 4 / (1 + x**2) dx */
  dx = 1. / n;
  sum = 0.0;
#pragma omp parallel for
  for (i = 0; i < n; i++) {
    x = 1. * i * dx;
    sum = sum + f(x);
  }
  pi = dx * sum;



#ifdef _OPENMP
  auto omp_elapsed = omp_get_wtime() - omp_t1;
#endif
  second elapsed = clk::now() - t1;


  std::printf("computed pi                     = %.16g\n", pi);
#ifdef _OPENMP
  std::printf("wall clock time (omp_get_wtime) = %.4gs in %d threads\n", omp_elapsed, nthreads);
#endif
  std::printf("wall clock time (chrono)        = %.4gs\n", elapsed.count());

  for(int d = 1; d <= 15; ++d) {
    std::printf("%d", digit(pi, d));
  }

  return 0;
}
```

We also have the following `pi_critical_correct.cc` file :

```
#include <chrono>
#include <cmath>
#include <cstdio>

#if defined(_OPENMP)
#include <omp.h>
#endif

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

inline int digit(double x, int n) {
  return std::trunc(x * std::pow(10., n)) -
         std::trunc(x * std::pow(10., n - 1)) * 10.;
}

inline double f(double a) { return (4. / (1. + a * a)); }

const int n = 10000000;

int main(int /* argc */, char ** /* argv */) {
  int i;
  double dx, x, sum, lsum, pi;

#if defined(_OPENMP)
  int num_threads = omp_get_max_threads();
#endif

#if defined(_OPENMP)
  auto omp_t1 = omp_get_wtime();
#endif
  auto t1 = clk::now();

  sum = 0.;
#pragma omp parallel shared(sum) private(x, lsum)
  {
    /* calculate pi = integral [0..1] 4 / (1 + x**2) dx */
    dx = 1. / n;
    lsum = 0.0;
#pragma omp for
    for (i = 1; i <= n; i++) {
      x = (1. * i - 0.5) * dx;
      lsum = lsum + f(x);
    }

#pragma omp critical
    sum += lsum;
  }

  pi = dx * sum;

#if defined(_OPENMP)
  auto omp_elapsed = omp_get_wtime() - omp_t1;
#endif
  second elapsed = clk::now() - t1;

  std::printf("computed pi                     = %.16g\n", pi);
#if defined(_OPENMP)
  std::printf("wall clock time (omp_get_wtime) = %.4gs on %d threads\n",
              omp_elapsed, num_threads);
#endif
  std::printf("wall clock time (chrono)        = %.4gs\n", elapsed.count());

  for (int d = 1; d <= 15; ++d) {
    std::printf("%d", digit(pi, d));
  }

  return 0;
}
```
And the following Makefile :

```
OPTIM+=-O3 -march=native -fopenmp
CXX=g++
CC=g++
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -std=c++11 $(OPTIM)
LDFLAGS+=-lm

EXE=pi_for_wrong pi_critical pi_critical_correct pi_reduction

all: clean $(EXE)

clean:
	rm -f $(EXE) *.o *~
```
