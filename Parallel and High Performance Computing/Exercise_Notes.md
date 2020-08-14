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
The corresponding `double_buffer.hh` file :

```
#ifndef DOUBLE_BUFFER
#define DOUBLE_BUFFER
#include <memory>
#include "grid.hh"

class DoubleBuffer {
public:
  DoubleBuffer(int m, int n);
  
  // defining the private and the public methods and parameters of the class
  Grid & current();
  Grid & old();

  void swap();
private:
  // this is a parameter not a method definition
  std::unique_ptr<Grid> m_current;
  std::unique_ptr<Grid> m_old;
};

#endif /* DOUBLE_BUFFER */
```

Then we have the following `dumpers.cc` file.

```
#include "dumpers.hh"
#include "grid.hh"
#include <iomanip>
#include <sstream>
#include <fstream>
#include <array>

// specifying the definitions of the methods by specifying the class then two two-dots
void Dumper::set_min(float min) {
  m_min = min;
}

void Dumper::set_max(float max) {
  m_max = max;
}

float Dumper::min() const {
  return m_min;
}

float Dumper::max() const {
  return m_max;
}

void DumperASCII::dump(int step) {
  // this is an output file stream
  std::ofstream fout;
  // probably means we only output strings 
  std::stringstream sfilename;

  // sets the width of the output to 5 
  sfilename << "output/out_" << std::setfill('0') << std::setw(5) << step << ".pgm";
  // the str() probably converts the passed sfilename to a string 
  // Opens the file identified by argument filename, associating it with the stream object,
  // so that input/output operations are performed on its content. Argument mode specifies the opening mode.
  fout.open(sfilename.str());

  int m = m_grid.m();
  int n = m_grid.n();

  fout <<  "P2" << std::endl << "# CREATOR: Poisson program" << std::endl;
  fout << m << " " << n << std::endl;
  fout << 255 << std::endl;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int v = 255. * (m_grid(i, j) - m_min) / (m_max - m_min);
      // you take the minimum of the variable and the value
      v = std::min(v, 255);
      fout << v << std::endl;
    }
  }
}

void DumperBinary::dump(int step) {
  std::ofstream fout;
  std::stringstream sfilename;

  sfilename << "out_" << std::setfill('0') << std::setw(5) << step << ".bmp";
  // Flags describing the requested input/output mode for the file.
  fout.open(sfilename.str(), std::ios_base::binary);

  int h = m_grid.m();
  int w = m_grid.n();

  int row_size = 3 * w;
  // if the file width (3*w) is not a multiple of 4 adds enough bytes to make it a multiple of 4
  // where you take modulo 4, or you find the remainder upon the division by four
  int padding = (4 - (row_size) % 4) % 4;
  row_size += padding;

  int filesize = 54 + (row_size)*h;
  // in this vector we have variables of type char only 
  std::vector<char> img(row_size*h);
  // Assigns the given value to the elements in the range [first, last).
  std::fill(img.begin(), img.end(), 0);

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      // the value of this float is given by the following calculation
      float v = ((m_grid(h - 1 - i, j) - m_min) / (m_max - m_min));

      float r = v * 255; // Red channel
      float g = v * 255; // Green channel
      float b = v * 255; // Red channel

      r = std::min(r, 255.f);
      g = std::min(g, 255.f);
      b = std::min(b, 255.f);

      img[row_size * i + 3 * j + 2] = r;
      img[row_size * i + 3 * j + 1] = g;
      img[row_size * i + 3 * j + 0] = b;
    }
  }
  // we have an array of characters of 14 elements 
  std::array<char, 14> bmpfileheader = {'B', 'M', 0, 0,  0, 0, 0,
                                        0,   0,   0, 54, 0, 0, 0};
  std::array<char, 40> bmpinfoheader = {40, 0, 0, 0, 0, 0, 0,  0,
                                        0,  0, 0, 0, 1, 0, 24, 0};

  bmpfileheader[2] = filesize;
  // The >> operator shifts its left-hand operand right by the number of bits 
  // defined by its right-hand operand.
  bmpfileheader[3] = filesize >> 8;
  bmpfileheader[4] = filesize >> 16;
  bmpfileheader[5] = filesize >> 24;

  bmpinfoheader[4]  = w;
  bmpinfoheader[5]  = w >> 8;
  bmpinfoheader[6]  = w >> 16;
  bmpinfoheader[7]  = w >> 24;
  bmpinfoheader[8]  = h;
  bmpinfoheader[9]  = h >> 8;
  bmpinfoheader[10] = h >> 16;
  bmpinfoheader[11] = h >> 24;
  bmpinfoheader[20] = (filesize - 54);
  bmpinfoheader[21] = (filesize - 54) >> 8;
  bmpinfoheader[22] = (filesize - 54) >> 16;
  bmpinfoheader[23] = (filesize - 54) >> 24;
  
  // Inserts the first n characters of the array pointed by s into the stream.
  // use data() method to get the data
  fout.write(bmpfileheader.data(), 14);
  fout.write(bmpinfoheader.data(), 40);

  fout.write(img.data(), h * row_size);
}
```

Where the corresponding `dumpers.hh` file is the following :

```
#ifndef DUMPERS_HH
#define DUMPERS_HH

class Grid;

class Dumper {
public:
  // looks like after the dots we are initializing the protected parameters to a certain value
  // typically here the m_grid is initialized to the grid reference passed in the constructor
  // then we have the curly brackets where the definition of the method should be
  explicit Dumper(const Grid & grid) : m_grid(grid), m_min(-1.), m_max(1.) {}
  
  // A virtual function is a member function which is declared within a base class and is 
  // re-defined(Overriden) by a derived class. When you refer to a derived class object using 
  // a pointer or a reference to the base class, you can call a virtual function for that object
  // and execute the derived class’s version of the function.
  // It's a pure virtual function. It makes it so you MUST derive a class (and implement said function) in order to use it.
  // https://www.geeksforgeeks.org/virtual-function-cpp/
  virtual void dump(int step) = 0;
  
  // the return type is void, which means nothing is returned 
  void set_min(float min);
  void set_max(float max);
  
  // the current instance of this is then constant which means that this function can not change 
  // the protected member variables 
  float min() const;
  float max() const;

protected:
  // The const keyword specifies that a variable's value is constant and tells the 
  // compiler to prevent the programmer from modifying it.
  const Grid & m_grid;
  // several floats are defined and separated by a comma 
  float m_min, m_max;
};

// deriving the base class which in this case is called Dumper 
class DumperASCII : public Dumper {
public:
  // Prefixing the explicit keyword to the constructor prevents the compiler 
  // from using that constructor for implicit conversions
  // calling the constructor of the base class after the two dots 
  explicit DumperASCII(const Grid & grid) : Dumper(grid) {}

  virtual void dump(int step);
};

class DumperBinary : public Dumper {
public:
  explicit DumperBinary(const Grid & grid) : Dumper(grid) {}

  virtual void dump(int step);
};

#endif /* DUMPERS_HH */
```

Then we have the following `grid.cc` file :

```
// you need to include the corresponding header .hh file for this .cc file 
#include "grid.hh"
#include <algorithm>

// within this method you call the other method
Grid::Grid(int m, int n) : m_m(m), m_n(n), m_storage(m * n) { clear(); }
void Grid::clear() { std::fill(m_storage.begin(), m_storage.end(), 0.); }

// these methods do not change the current instant of the class so they are const
// methods and they return an int
int Grid::m() const { return m_m; }
int Grid::n() const { return m_n; }
```

Then we have the following `grid.hh` file :

```
#ifndef GRID_HH
#define GRID_HH
#include <vector>

class Grid {
public:
  Grid(int m, int n);

  // access the value [i][j] of the grid
  // returns a reference to a float 
  inline float & operator()(int i, int j) { return m_storage[i * m_n + j]; }
  // returns a float that we should not be able to modify later
  // but also the method itself does not modify the current instant
  inline const float & operator()(int i, int j) const {
    return m_storage[i * m_n + j];
  }

  // set the grid to 0
  void clear();

  int m() const;
  int n() const;

private:
  int m_m, m_n;
  std::vector<float> m_storage;
};

#endif /* GRID_HH */  
```

Then we have the following `poisson.cc` file :

```
#include "simulation.hh"
#include <chrono>
#include <iostream>
#include <sstream>
#include <tuple>
#include <chrono>
#include <omp.h>
// where we define here some constants
#define EPSILON 0.005

typedef std::chrono::high_resolution_clock clk;
typedef std::chrono::duration<double> second;

// looks like in the .cc files you need to refer to strings through the std:: namespace
static void usage(const std::string & prog_name) {
  // when you output an error
  std::cerr << prog_name << " <grid_size>" << std::endl;
  exit(0);
}

// argv and argc are how command line arguments are passed to main() in C and C++.
// The variables are named argc (argument count) and argv (argument vector)
int main(int argc, char * argv[]) {
  // apply the usage method to the first element input into the prompt
  if (argc != 2) usage(argv[0]);
  
  std::stringstream args(argv[1]);
  int N;
  // This is used to read from stringstream object.
  // meaning the value is extracted into the integer N
  args >> N;

  if(args.fail()) usage(argv[0]);
  
  // in c++ unlike in Java you do not write new Object, you just directly write
  // in the corresponding arguments
  Simulation simu(N, N);

  simu.set_initial_conditions();

  simu.set_epsilon(EPSILON);

  float l2;
  int k;

  auto start = clk::now();
  // std::tie constructs and returns a tuple of references.
  std::tie(l2, k) = simu.compute();
  auto end = clk::now();

  second time = end - start;

  std::cout << omp_get_max_threads() << " " << N << " "
            << k << " " << std::scientific << l2 << " "
            << time.count() << std::endl;

  return 0;
}
```
