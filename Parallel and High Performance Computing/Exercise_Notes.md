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

