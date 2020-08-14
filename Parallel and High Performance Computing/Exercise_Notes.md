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
// The elements in this header deal with time. 
#include <chrono>
#include <cmath>
#include <cstdio>

// If the /openmp flag was passed to the compiler and compilation was successful
// a preprocessor directive will be added that you can use to check this at runtime:

#if defined(_OPENMP)
#include <omp.h>
#endif

// Class std::chrono::high_resolution_clock represents the clock with the 
// smallest tick period provided by the implementation.
using clk = std::chrono::high_resolution_clock;
// It consists of a count of ticks of type Rep and a tick period, where the tick period 
// is a compile-time rational constant representing the number of seconds from one tick to the next.
using second = std::chrono::duration<double>;
// Class template std::chrono::time_point represents a point in time. 
// It is implemented as if it stores a value of type Duration indicating the time interval from the start of the Clock's epoch.
using time_point = std::chrono::time_point<clk>;

inline int digit(double x, int n) {
  // Computes the nearest integer not greater in magnitude than arg for std::trunc
  return std::trunc(x * std::pow(10., n)) -
         std::trunc(x * std::pow(10., n - 1)) * 10.;
}

// C++ inline function is powerful concept that is commonly used with classes. 
// If a function is inline, the compiler places a copy of the code of that 
// function at each point where the function is called at compile time.

// Any change to an inline function could require all clients of the function to be 
// recompiled because compiler would need to replace all the code once again 
// otherwise it will continue with old functionality.

inline double f(double a) { return (4. / (1. + a * a)); }

const int n = 10000000;

// didn't find the meaning of the below parameters in the main function
int main(int /* argc */, char ** /* argv */) {
  int i;
  double dx, x, sum, pi;

#if defined(_OPENMP)
  int num_threads = omp_get_max_threads();
#endif

#if defined(_OPENMP)
  // the compiler deduces the type of the variable 
  auto omp_t1 = omp_get_wtime();
#endif
  auto t1 = clk::now();

  sum = 0.;
#pragma omp parallel shared(sum) private(x)
  {
    // the code to be parallelized is written within these curly brackets
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
  // The omp_get_wtime routine returns elapsed wall clock time in seconds.
  auto omp_elapsed = omp_get_wtime() - omp_t1;
#endif
  second elapsed = clk::now() - t1;
  // %[flags][width][.precision][length]specifier
  // http://www.cplusplus.com/reference/cstdio/printf/
  std::printf("computed pi = %.16g\n", pi);
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

Then we also have the `pi_for_wrong.cc` and `pi_critical_correct.cc` files, but these ressemble a lot the one above so we don't include it here. The following is the Makefile :

```
OPTIM+=-O3 -march=native -fopenmp
CXX=g++
CC=g++
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -std=c++11 $(OPTIM)
LDFLAGS+=-lm

// we have this list of files in the folder 
EXE=pi_for_wrong pi_critical pi_critical_correct pi_reduction

// https://stackoverflow.com/questions/2635453/how-to-include-clean-target-in-makefile
all: clean $(EXE)

// the clean keyword means that during compilation the following files are deleted 
// that have a .o extension or a ~ ending or have an EXE file name, before the output files
// are generated once again 
clean:
	rm -f $(EXE) *.o *~
```

Now consider the *poisson* folder. We have the following code inside. The Makefile :

```
CXX=g++
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -Werror -pedantic -std=c++11 -fopenmp -O3 -march=native
LDFLAGS+=-lm -fopenmp

// all the cc files have then a corresponding .o file
OBJS=poisson.o simulation.o double_buffer.o grid.o dumpers.o

all: poisson

// where poisson is defined as below and the dollar sign signifies a variable 
// using the g++ compiler 
poisson: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f hello poisson *.o *~
```

The `double_buffer.cc` file :

```
// where here you can include the header files in double quotes
#include "double_buffer.hh"
#include "grid.hh"

// where the DoubleBuffer is a class which is defined in the header file double_buffer.hh
// where we must be calling here the constructor of that class
// and we define the constructor as having an empty definition, but also you
// define m_current as being a new Grid 
// std::unique_ptr<MyClass> my_p_obj( new MyClass(myObject) );
// https://stackoverflow.com/questions/37180818/c-how-to-convert-already-created-object-to-unique-ptr
DoubleBuffer::DoubleBuffer(int m, int n)
    : m_current(new Grid(m, n)), m_old(new Grid(m, n)) {}

// the following is the definition of the public methods which do not need an instantiation 
// of the class DoubleBuffer in order to be defined 
// here we are reeturning a pointer to the private method 
// The return type is a reference to a grid
Grid & DoubleBuffer::current() { return *m_current; }
Grid & DoubleBuffer::old() { return *m_old; }

// the definition of the last public method 
void DoubleBuffer::swap() {
  // you can access the swap method of the unique pointer to the grid 
  m_current.swap(m_old);
}
```
