# Parallel and High Performance Computing

**Week 1**

The Von Neumann architecture consists in an ALU (an arithmetical and logical unit) along with a control unit in a central processing unit together. There is a memory unit linked to the ALU, as well as input and output devices. The system bus consists in the control, the address and the data bus. The address bus gives us instructions on where to find the data, the data bus gives instructions on what to find, the control bus gives instructions on what to do with the data (for example : read, write, fetch, etc...). The CPU performance is measured in FLOPS. The memory bandwidth in bytes per second and the memory latency in seconds. The peak values are calculated as follows :

Peak CPU performance = CPU Frequency X Number of Operations per clock cycle X size of the largest vector X number of cores
Memory bandwidth = RAM Frequency X Number of transfers per clock cycle X Bus width X number of interfaces
Memory latency depends on the size of the data. Usualy given by the constructor

There are limits to how fast a CPU can run and its circuitry cannot always keep up with an overclocked speed. If the clock tells the CPU to execute instructions too quickly, the processing will not be completed before the next instruction is carried out. If the CPU cannot keep up with the pace of the clock, the data is corrupted. CPUs can also overheat if they are forced to work faster than they were designed to work. (https://www.bbc.co.uk/bitesize/guides/zmb9mp3/revision/2)

Memory bandwidth and latency are key considerations in almost all applications, but especially so for GPU applications. Bandwidth refers to the amount of data that can be moved to or from a given destination. In the GPU case we’re concerned primarily about the global memory bandwidth. Latency refers to the time the operation takes to complete.

The High Performance Linpack seems to be a test where an n x n system of linear equations is solved using Gauss pivoting. The machine code is the only language understandable by the processor.

In the computer the gcc -E command during the preprocessing step outputs a .i files where the code remains the same and the hashtags at the top of the file are different. Then the gcc -S command transforms the .i file into a .s file, this .s file has the commands written as pushq, movq, movl etc. Then the assembler transforms the .s file into a .o file which has a less understandable syntax. The linker then produces the actual executable (by linking against the external libraries if required). We can execute two .c files into one executable as follows :

```
gcc file1.c file2.c -o app.exe
./app.exe
```

Nodes are composed of one or more CPI each with multiple cores, a shared memory among cores, a Networking Interface Card NIC. There is an interconnection network, one or more frontend to access the cluster remotely. These are all the hardware components. Now what concerns the software components we have a scheduler in order to share the resources among multiple users, a basic software stack with compilers, linkers and basic parallel libraries, and an application stack with specific read-to-use scientific applications. The Flynn taxonomy has different names for the different categories which can result from processing single or multiple data, againstsingle or multiple instructions. We have SISD, sequential processing, SIMD on vector processing, and MIMD for distributed processing. You can have data parallelism, when the same instruction is executed on different data as in SIMD, or you can have task parallelism, when multiple different instructions are executed in parallel as is the case in MISD and MIMD. 

In data parallelism the data resides on a global memory address space. In task parallelism, each task is excuted on a specialized piece of harwdware and all this is done concurrently. We need a Message Passing API, called MPI, or Message Passing Interface, in HPL, High Performance Linpack. The speedup and efficiency can be calculated as follows where T_1 is the execution time using 1 process, T_p is the execution time using p processes, S(p) is the speedup of p processes and E(p) is the parallel efficiency.

```
S(p) = T_1 / T_p

E(p) = S(p) / p
```

There is a maximal achievable speedup when a fraction f of the code can not be parallelized, then we have that : 

```
S(p) <= p/(1 + (p-1)f)
```

We have that even as the number of nodes p tends to infinity, then S(p) will still be smaller than or equal to 1/f. Suppose that we do achieve the maximum speedup then we have the following equality for the fraction of code that is not parallelizable :

<a href="https://www.codecogs.com/eqnedit.php?latex=f&space;=&space;\frac{\frac{1}{S_p}&space;-&space;\frac{1}{p}}{1&space;-&space;\frac{1}{p}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f&space;=&space;\frac{\frac{1}{S_p}&space;-&space;\frac{1}{p}}{1&space;-&space;\frac{1}{p}}" title="f = \frac{\frac{1}{S_p} - \frac{1}{p}}{1 - \frac{1}{p}}" /></a>

Suppose s is the sequential portion, a is the parallel porion, then the sequential t_s and the parallel t_p values are given by :

<a href="https://www.codecogs.com/eqnedit.php?latex=t_s&space;=&space;s&space;&plus;&space;an" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t_s&space;=&space;s&space;&plus;&space;an" title="t_s = s + an" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=t_p&space;=&space;s&space;&plus;&space;\frac{an}{p}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t_p&space;=&space;s&space;&plus;&space;\frac{an}{p}" title="t_p = s + \frac{an}{p}" /></a>

Here n is the size of the problem. Then the speedup is :

<a href="https://www.codecogs.com/eqnedit.php?latex=s_p&space;=&space;\frac{t_s}{t_p}&space;=&space;\frac{\frac{s}{n}&space;&plus;&space;a}{\frac{s}{n}&space;&plus;&space;\frac{a}{p}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_p&space;=&space;\frac{t_s}{t_p}&space;=&space;\frac{\frac{s}{n}&space;&plus;&space;a}{\frac{s}{n}&space;&plus;&space;\frac{a}{p}}" title="s_p = \frac{t_s}{t_p} = \frac{\frac{s}{n} + a}{\frac{s}{n} + \frac{a}{p}}" /></a>

In theoretical computer science, communication complexity studies the amount of communication required to solve a problem when the input to the problem is distributed among two or more parties. In a task dependency graph, the nodes are the tasks and the edges are the dependencies. The critical path is the longest path from the starting task to the ending task. The average degree of concurrency is the total amount of work divided by the critical path length.

**Week 2**

A cluster is composed of computer nodes that form the back-end, there is a scheduler which connected to the back-end, and login nodes which form the front-end. The login nodes are connected to the scheduler and the computer nodes. The computer nodes are connected to the cluster filesystem and the shared filesystems. To connect to a remote cluster you probably need to this via internet ? You can use modules using the following commands :

```
module avail
module load <module>
module unload <module>
module purge 
module list
```

The makefile is a build automation system, it is a set of rules on how to produce an executable, what to compile, how to link, what to link, etc. There are two types of libraries : static and shared. Static libraries are archives of object files, shared libraries are libraries loaded dynamically at execution. To generate a shared library :

```
g++ -o filename.so -shared filename.o
g++ -L<path to the library> -name -o <executable> <object_1> ... <object_n>
```

The job is submitted from the login nodes to the scheduler, which submits the job to the computer nodes, these run the job. SLURM can be written as Simple Linux Utility for Resource Management, and it is a job scheduler. Some basic commands are as follows :

```
sbatch : submit a job to the queue
salloc : allocated resources
squeue : visualize the state of the queue
```
The common options of SLURM are :

```
-A --account : defines which account to use
-u --user : defines which user to use, useful for squeue
--reservation : defined which reservation to use
--p --partition : defines which partition to use, list of partitions can be found with sinfo
-N --nodes : defines the number of nodes to use
-t --time : defines the maximum wall time
-n --tasks : number of tasks in the MPI sense
-c --cpus-per-task : number of cpus per process
--ntasks-per-node : number of tasks per node 
--mem : defines the quantity of memory per node requested
```

Now we study Git. We can clone as follows :

```
git clone <uri repo.git>
git status
git add <filename>
git commit -m <message>
git push
git pull
```

When you are working on the same file from two different devices, to commit the changes, first pull then type `git commit -a`, then push the modifications. In git you can checkout branches as follows : `git checkout -b feature`, where here feature is the name of the copy you checked out of the master repo. Now when you commit you commit to this feature copy. Then when you want to merge the branch with the master you can write : `git merge feature`. Often you could decide to have multiple servers. You can write :

```
# on the second remote server
git init --bare
# on the computer
git remote add server2 <remote url 2>
git push server2 
```

**Week 3**

Latency: time to complete a operation. Throughput: how many operations per time. These two concepts are related with Little's law, which says that the average number of items in the queuing system is the average number of items arriving per unit time times the average waiting time in the system for an item. If we apply this concept to CPUs then we have that latency is how long it takes before the next dependent operation can start, and the throughput is the number of independent operations per time unit. We have that the measure of parallelism is given by the product of the latency and the throughput. We have some principles related to the HPC. If memory bandwidth goes up, the latency does not go down. You can increase the throughput by maximizing locality. The CPU needs to know where to fetch the data : contiguous accesses will maximize the throughput, non-contiguous accesses are latency bound. We have that all of the following seem to increase over time : transistors, parallel processors perormance, sequential processor performance, frequency, typical power, number of cores. There are different levels of performance such as distributed parallelism, thread level parallelism, data level parallelism, instruction level parallelism. Pipelining says that like the Ford’s assembly line, instructions are broken down in many small steps (stages). There is increased IPC through increased parallelism, smaller stages means increased frequency which unlocked the frequency era. There are multiple pipelines to increase instructions per cycles. This can be spoiled by data, control, structural hazards and multi-cycle instructions. 

Speculative execution is tentative execution despite dependencies. Thread-level parallelism uses multiple concurrent threads of execution that are inherently parallel. Increase throughput of multithreaded codes by covering the latencies. The goal of HPC is to increse mathematical throughput : Latency is NOT going down therefore throughput is increased, throughput is going up IF parallelism is increased, avoid pipeline stalls by having data “close” to the CPU. HPC kernel optimization focus on extracting parallelism and maximizing data locality. Kernels can be represented by the number of mathematical operations, or the number of data transfers. We have AI = flops/DRAM accesses. We have the following equation to specify the peak FP performance theoretically :

Peak FP performance = Number of FP ports * flops/cycles * vector size * frequency * number of cores

The Roofline Model has advantages and disadvantages. The advantages caps the performance, allows visual goals for optimization, easily used everywhere. The disadvantages are that latencies need to be covered, there is an oversimplification, node-only, "vanilla" version is cache-oblivious. In the floating point representation of the numbers we have a number can be represented as :

<a href="https://www.codecogs.com/eqnedit.php?latex=(-1)^s&space;(d_0&space;&plus;&space;d_1&space;\beta&space;&plus;&space;d_2&space;\beta^2&space;&plus;&space;...&space;&plus;&space;d_{p-1}&space;\beta^{p-1})&space;\beta&space;^e" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(-1)^s&space;(d_0&space;&plus;&space;d_1&space;\beta&space;&plus;&space;d_2&space;\beta^2&space;&plus;&space;...&space;&plus;&space;d_{p-1}&space;\beta^{p-1})&space;\beta&space;^e" title="(-1)^s (d_0 + d_1 \beta + d_2 \beta^2 + ... + d_{p-1} \beta^{p-1}) \beta ^e" /></a>

Here beta is the base, e is the exponent and s is the sign.

**Week 4**

Here we study optimization on one core, OpenMP, MPI and hybrid programming. The cycle of software design looks as follows : requirement analysis, design, implementation, testing, evolution. When you have no serial code, you design your application in a parallel way from scratch. When you have a serial code you follow a debugging-profiling-optimization cycle before any parallelization. You can debug using standard tools like gdb, or new tools such as Totalview, Alinea, DDT or recently Eclipse PTP.

We can profile with gprof and Intel Amplifier. With this we see the Conjugate Gradient solver is the most consuming part, the CPI value (Clockticks per Instructions Retired) is good. You can optimize using compiler and linker flags, optimized external libraries, handmade refactoring, algorithmic changes. You can parallelize only when the sequential code has no bugs and is optimized. You can ask yourself the following questions while parallelizing : is it worth to parallelize the code, does the algorithm scale, what are the performance predictions, will there be bottlenecks, which parallel paradigms should we use, what is the target architecture ? Flop is the Floating Point Operations per Second. It's the theoretical number of operations divided by the running time. You can measure memeory characteristics by hand (the theoretical number of access divided by the running time), with external tools (all based on hardware counters). A thread is an execution entity with a stack and a static memory. Thread-safe routine is a routine that can be executed concurrently. The memory model is such that there is a shared memory, threads connected to the memory and the private memory connected to the thread. All what you can do with OpenMP can be done with MPI and/or threads. We have BASH-like shells `export OMP_NUM_THREADS=4`, and in the CSH-like shells we have `setenv OMP_NUM_THREADS=4`. Compiler directives that allow work sharing, synchronization and data scoping. There is runtime library that contains informal, data access and synchronization directives. To write a parallel construct in OpenMP we can write :

```
#pragma omp parallel [clause[[,] clause]...]
{
  structured-block
}
```

Where clause is one of the following : if or num_threads which is the conditional clause, default(private | firstprivate | shared | none ) is the default data scoping, private(list), firstprivate(list), shared(list) or copyin(list) is the data scoping, or it could be reduction({ operator | intrinsic_procedure_name } : list). What is data scoping ? We determine which variables are private to a thread, which are shared among all threads. The default scope is shared, this is the most difficult part of OpenMP. The private attribute means the data is private to each thread and non-initialized. We have `#pragma omp parallel private(i)`. The shared attribute is used to denote that the data is shared among all the threads, an example is `#pragma omp parallel shared(array)`. The firstprivate is like private but it is initialized to the value before the parallel region, the lastprivate region is like private but the value is updated after the parallel region. Worksharing constructs are possible with the following keywords `sections`, `single` and `workshare` only for Fortran. Only one thread executes the single region, the others wait for completion, except if the nowait clause has been activated. Suppose you want to parallelize the following for loop. Then we can write :

```
#pragma omp for [clause[[,] clause] ...]
{
  foor-loop
}
```
Where clause is one of the following : `schedule(kind[, chunk_size}), collapse(n), ordered, private(list), firstprivate(list), lastprivate(list), reduction()`. Consider the different types of kind keywords in the schedule. Suppose kind is static, then the iterations are divided into chunks sized chunk_size assigned to threads in a round-robin fashion. If the chunk-size is not specified, the system decides. Suppose that the kind is dynamic, the iterations are divided in chunks sized chunk_size assigned to threads when they request them until no chunk remains to be distributed. If the chunk_size is not specified, then the default is one. Suppose that the kind is guided, then the iterations are divided in chunks sized chunk_size assigned to threads when they request them. The size of the chunks is proportional to the remaining unassigned chunks. By default the chunk size is approximately loop_count/number_of_threads. The kind auto specifies that the decisions is delegated to the compiler and/or the runtime system. The kind runtime specifies that the decision is delegated to the runtime system. Here we have a parallel for example :

```
#pragma omp parallel shared(A, B, C) private(i,j,k, myrank)
{
  myrank = omp_get_thread_num();
  mysize = omp_get_num_threads();
  chunk = (N/mysize);
  #pragma omp for schedule(static, chunk)
  for (i=0; i<N; i++){
    for (j=0; j<N; j++){
      for (k=0; k<N; k++){
        C[i][j] = C[i][j] + A[i][k]*B[k][j];
      }
    }
  }
}
```

Use the collapse clause to increase the total number of iterations
that will be partitioned across the available number of OMP
threads by reducing the granularity of work to be done by each
thread. You can improve performance by avoiding use of the collapsed-loop
indices (if possible) inside the collapse loop-nest (since the
compiler has to recreate them from the collapsed loop-indices
using divide/mod operations AND the uses are complicated
enough that they don’t get dead-code-eliminated as part of
compiler optimizations). A collapse directive example :

```
#pragma omp parallel for collapse(2) shared(A) private(k,l)
  for(k=0; k<kmax;k++) {
    for(l=0; l<lmax; l++) {
      for(i=0; i<N; i++) {
        for(j=0; j<N;j++) {
          A[i][j] = A[i][j]*s + A[i][j]*t;
        }
      }
    }
  }
```

What is an OpenMP task ? It offers a solution to parallelize irregular problems (unbounded loops, recursives, master/slave schemes etc...). OpenMP tasks are composed of code that will be executed, data initialized at task creation time, ICV's which are Internal Control Variables. The sychronization says that all tasks  created by a thread of a team are guaranteed to be completed at thread exit. Within a task group, it is possible to synchronize through #pragma omp taskwait.

The task directive has the following execution model. A task t is executed by the thread T of the team that generated it. A thread T can suspend/resume/restart a task t. Tasks are tied by default : these are executed by the same thread, tied tasks hav scheduled restrictions. It is possible to untie tasks using the directive untied. We have the following synchronization constructs. The following directives are mandatory : mandatory which means the region is executed by the master thread only, critical which means that the region is executed by only one thread at a time, barrier which means all threads must reach this directive to continue, taskwait which means that all tasks and childs must reach this directive to continue, atomic(read|write|update|capture) which means the asociated storage location is accessed by only one thread/task at a time, flush which means that this operation makes the thread's temporary view of memory consistent with the shared memory, ordered which means that a structured block is executed in order of the loop iterations. An example of the master construct :

```
#pragma omp parallel default(shared)
{
  #pragma omp master
  {
    printf("I am the master\n");
  }
}
```

You can nest parallel regions within other parallel regions. You can include the libomp.so and libgomp.so by writing `#include <omp.h>`. We have the following runtime library routines : `omp_get_wtime` meaning it returns the elapsed wall clock time in seconds, `omp_get_wtick` meaning it returns the precision of the timer used by `omp_get_wtime`. You can set the environment variables as follows. Under csh we have `setenv OMP_VARIABLE "value"`, and under bash we have `export OMP_VARIABLE="value"`. The OMP_SCHEDULE variable sets the run-sched-var ICV that specifies the runtime schedule type and chunk size. It can be set to any of the valid OpenMP schedule types. The OMP_NUM_THREADS sets the nthreads-var ICV that specifies the number of threads to use in parallel regions. Affinity is a measure which measures on which core does the thread run. In Intel Executable we can write as follows `KMP_AFFINITY = verbose, SCHEDULING` you are able to see where the OS pin each thread. You can show and set the affinity with the GNU executable by setting the export `GOMP_CPU_AFFINITY = verbose, SCHEDULING`, this way you are able to see where the OS pin each thread. Here `SCHEDULING` can be `scatter` or `compact`. The new version OpenMP 4.0 has support for new devices (Intel, Phi, GPU...) with `omp target`. We can specify a league of threads with `omp teams` and distribute a loop over the team with `omp distribute`. There is SIMD support for vectorization `omp simd`. You can set the thread affinity with a more standard way than `KMP_AFFINITY` with the concept of `places` (a thread, a core, a socket), `policies` (spread, close, master) and `control settings` the new clause `proc_bind`.

**Week 5**

We have optimization on one core, OpenMP, MPI and hybrid programming. We have an overview of the de-facto industrial standards. We will study the distributed memory programming paradigm MPI. The goal of MPI is to provide a source-code portability. MPI is used to run multiple instances of the same program, for example the following code `mpirun -np p myApp myArgs` starts p instances of the program `myApp myArgs`. Instances exchange information by sending messages to each other. Communications take place within a communicator : a set of processes indexed from 0 to the communicatorSize - 1. A special communicator called MPI_COMM_WORLD contains all the processes. There are different types of communication in MPI, the following : point-to-point, collectives, one-sided. We have the following methods :

```
MPI_Send(buf, count, datatype, destination, tag, communicator)
MPI_Recv(buf, count, datatype, source,tag, communicator, status)
```

Each send must be matched by a receive. Here is an example :

```
int main(int argc, char *argv[]) {
  int rank, size;
  int buf[100];
  MPI_Status status;
  // initiallized with the references to the command line input number and actual values
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  if (rank == 0) {
   MPI_Send(buf, 100, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (rank==1) {
   MPI_Recv(buf, 100, MPI_INT, 0, 0, MPI_COMM_WORLD,&status);
  }
  MPI_Finalize();
}
```

We have the following blocking point-to-point communication keywords that we can use. `MPI_Send` returns when buffer can be reused, `MPI_Ssend` returns when the other end posted matchs recv, `MPI_Recv` returns when the message has been received, `MPI_Sendrecv` send and receives within the same call to avoid deadlocks, `MPI_Bsend` returns immediately and send a buffer than can be reused immediately, `MPI_Rsend` returns only when the send buffer can be safely reused, `MPI_Sendrecv_replace` which can send, receive and replaces the buffer values using only one buffer.

We have the following non-blocking point-to-point comunications keywords that we can use. `MPI_Isend` and `MPI_Irecv` do not wait for message to be buffered send/recv. It fills an additional `MPI_Request` parameter that identifies the request. The following wait calls block until the requests are completed : `MPI_Wait(request, status), MPI_Waitall(count, array_of_requests, array_of_statusses)`. We have also non-blocking versions of the previous keywords seen in the paragraph above, `MPI_Issend`, `MPI_Ibsend`, `MPI_Irsend`. We have the following waiting and test keywords : `MPI_Waitsome` waits for an MPI request to complete, `MPI_Waitany` waits for any specifies MPI request to complete, `MPI_Test` tests for the completion of a request, `MPI_Testall` tests for the completion of all previously initiated requests, `MPI_Testany` tests for the completion of any previously initiated requests, `MPI_Testsome` tests for some given requests to complete. We have the following examples of code using MPI :

```
int main(int argc, char *argv[]) {
  int rank;
  int buf[100];
  MPI_Request request;
  MPI_Status status;
  // initialization using references to the argument count and the array of arguments
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    MPI_Isend(buf, 100, MPI_INT, 1, 0,MPI_COMM_WORLD, &request);
  } else if (rank == 1) {
    MPI_Irecv(buf, 100, MPI_INT, 0, 0,MPI_COMM_WORLD, &request);
  }
  // add references to the request and to the status
  MPI_Wait(&request, &status);
  MPI_Finalize();
}
```

Then we have the following code where process 0 and 1 exchange the content of their buffer with non-blocking actions :

```
if (rank == 0) {
  MPI_Isend(buf1, 10, MPI_INT, 1, 0, MPI_COMM_WORLD,&request);
  MPI_Recv(buf2, 10, MPI_INT, 1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
}else if (rank == 1){
  MPI_Isend(buf1, 10, MPI_INT, 0, 0, MPI_COMM_WORLD,request);
  MPI_Recv(buf2, 10, MPI_INT, 0, 0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
}

MPI_Wait(&request, &status);
// you copy from the first buffer to the second, the first number of elements specified
// by the third parameter
memcpy(buf1, buf2, 10*sizeof(int));
```

In the below processes 0 and 1 exchange the content of their buffers with `sendrecv`. 

```
if (rank == 0) {
  MPI_Sendrecv(buf1, 10, MPI_INT, 1, 0, buf2, 10,
              MPI_INT, 1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
} else if (rank == 1) {
  MPI_Sendrecv(buf1, 10, MPI_INT, 0, 0, buf2, 10,
      MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

memcpy(buf1, buf2, 10*sizeof(int));
```

You can also exchange the contents of the buffers from processes 0 and 1 with the keyword `sendrecv_replace()`. 

```
if (rank == 0){
  MPI_Sendrecv_replace(buf1, 10, MPI_INT, 1, 0, 1, 0, 
  MPI_COMM_WORLD,MPI_STATUS_IGNORE);
} else if (rank == 1) {
  MPI_Sendrecv_replace(buf1, 10, MPI_INT, 0, 0, 0, 0, 
  MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```

We have that `MPI_ANY_SOURCE` and `MPI_ANY_TAG` are wildcards. We have the following collective communications : `MPI_Bcast` sends the same data to every process, `MPI_Scatter` sends pieces of the buffer to every process of the communicator, `MPI_Gather` retrieves pieces of data from every process, `MPI_Allgather` all pieces retrieved by all processes, `MPI_Reduce` performs a reduction operation across all nodes, `MPI_Allreduce` the result is distributed to all processes, `MPI_Alltoall` sends all data to all processes, every process of the communicator must participate. The following code is for when you receive image parts out of order.

```
MPI_Isend(imgPart, partSize, MPI_BYTE, 0,0, MPI_COMM_WORLD, &request);
if (rank == 0) {
  char *buf = malloc(nProcs * partSize);
  MPI_Status s;
  int count;
  for (int i = 0; i < nProcs; i++) {
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &s);
    MPI_Get_count(&s, MPI_BYTE, &count);
    MPI_Recv(buf + s.MPI_SOURCE*count, count, MPI_BYTE, s.MPI_SOURCE, s.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
} MPI_WAIT(&request, MPI_STATUS_IGNORE);
```

We receive the image parts with a collective :

```
int root = 0;
char *buf = NULL;
if (rank == root) {
  buf = malloc(nProcs*partSize);
  MPI_Gather(part, partSize, MPI_BYTE, buf, partSize, MPI_BYTE, root, MPI_COMM_WORLD);
} else {
  MPI_Send(part, ..., ..., rank, MPI_TAG);
}
```

You can define your own communicators. `MPI_Comm_dup` duplicates a communicator (eg : to enable private communications within library functions), `MPI_Comm_split` splits a communicator into multiple smaller communicators (useful when using 2D and 3D domain decomposition). We time the MPI programs. `MPI_Wtime()` returns a double precision floating point number, the time in seconds since some arbitrary point of time in the past. `MPI_Wtick()` returns a double precision floating point number, the time in seconds between successive ticks of the clock.

There are also datatypes related to the MPI library. We have other derived MPI datatypes : `MPI_Type_contiguous` which produces a new data type by making copies of an existing data type, `MPI_Type_vector` which is similar to contiguous, but allows for regular gaps in the dispacements, `MPI_Type_indexed` is an array of displacements of the input data type which is provided as the map for the new data type, `MPI_Type_create_struct` is the new data type which is formed according to a completely defined map of the component data types, `MPI_Type_extent` returns the size in bytes of the specified data type, `MPI_Type_commit` commits the new datatype to the system, `MPI_Type_free` deallocates the specified datatype object.

```
struct { int a; char b; } foo;
MPI_Aint zero_address, first_address, second_address;
MPI_Get_address(&foo, &zero_address);
MPI_Get_address(&foo.a, &first_address);
MPI_Get_address(&foo.b, &second_address);
MPI_Datatype newtype;
MPI_Aint displs[2];
blen[0] = 1; 
indices[0] = MPI_Aint_diff(first_address, zero_address);
oldtypes[0] = MPI_INT; 
oldtypes[1] = MPI_CHAR;
blen[1] = 1; 
indices[1] = MPI_Aint_diff(second_address, first_address);
MPI_Type_create_struct( 2, blen, indices, oldtypes, &newtype );
MPI_Type_Commit(&newtype);
foo f = {1,’z’};
MPI_Send(&f, 1, newtype, 0, 100, MPI_COMM_WORLD );
MPI_Type_free( &newtype );
```

Now we want to pack and unpack the data using the following code :

```
int x; float a, int position=0;
char buffer[100];
if (myrank==0) {
  MPI_Pack(&a, 1, MPI_FLOAT, buffer, 100, &position,MPI_COMM_WORLD)
  MPI_Pack(&x, 1, MPI_INT, buffer, 100, &position,MPI_COMM_WORLD)
  MPI_Send(buffer, 100, MPI_PACKED, 1, 999,MPI_COMM_WORLD);
}else if (myrank==1) {
  MPI_Recv(buffer, 1000, MPI_PACKED, 0, 999,MPI_COMM_WORLD, status)
  MPI_Unpack(buffer, 100, &position, &a, 1,MPI_FLOAT, MPI_COMM_WORLD);
  MPI_Unpack(buffer, 100, &position, &x, 1, MPI_INT,MPI_COMM_WORLD);
}
```

An MPI group is an ordered set of processes, each process has a unique ID and can belong to several different groups. A group can be used to create a new communicator. An MPI communicator is a group of processes, it encapsulates the communications between the belonging processes, an MPI communication can take place only with a communicator. We have the following code to create a new communicator :

```
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_group(MPI_COMM_WORLD, &old_g);
int nbr_g1 = 5;
// is this like type casting 
ranks1 = (int*) malloc(nbr_g1*sizeof(int));
ranks2 = (int*) malloc((size-nbr_g1)*sizeof(int));
for (i=0;i<nbr_grp1;i++) ranks1[i]=i;
for (i=0;i<(size-nbr_g1);i++) ranks2[i]=size-i-1;
if (rank < nbr_g1) {
  MPI_Group_incl(old_g,nbr_g1,ranks1,&new_g);
} else {
  MPI_Group_incl(old_g,(size-nbr_g1),ranks2,&new_g);
}
MPI_Comm_create(MPI_COMM_WORLD,new_g,&new_comm);
MPI_Group_rank(new_g, &new_rank);
printf("rank %d grprank is %d \n",rank,new_rank);
MPI_Finalize();
```

We have persistent communications example : 

```
MPI_Request recvreq;
MPI_Request sendreq;

MPI_Recv_init (buffer, N, MPI_FLOAT, rank-1,tag_check_infos, MPI_COMM_WORLD, &recvreq);
MPI_Send_init (buffer, N, MPI_FLOAT, rank+1,tag_check_infos, MPI_COMM_WORLD, &sendreq);

/* ... copy stuff into buffer ... */

MPI_Start(&recvreq);
MPI_Wait(&recvreq, &status);
MPI_Request_free( &recvreq );
MPI_Request_free( &sendreq );
```

There is one-side communication keywords. For the initialization we have `MPI_Alloc_Mem(), MPI_Free_Mem(), MPI_Win_Create(), MPI_Win_Free()`. For remote memory access we have : `MPI_Put(), MPI_Get(), MPI_Accumulate()`. For the synchronization we have : `MPI_Win_Fence(), MPI_Win_Post(), MPI_Win_Start(), MPI_Win_Complete(), MPI_Win_Wait(), MPI_Win_Lock(), MPI_Win_Unlock()`. We have the following example :

```
int MPI_Alloc_mem(MPI_Aint size, MPI_Info info, void *baseptr);
int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win)
int MPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)
int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)
```

There is the one-sided communications example :

```
MPI_Win win;
int *mem;
float x = 1.0;
MPI_Alloc_mem(size * sizeof(int), MPI_INFO_NULL, &mem);
MPI_Win_create(mem, size * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

// Write x at position 1 within process 0 ’s memory
MPI_Put(&x, 1, MPI_FLOAT, 0, rank, 1, MPI_INT, win);
MPI_Win_free(win);
MPI_Free_mem(mem);
```

