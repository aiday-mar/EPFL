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
