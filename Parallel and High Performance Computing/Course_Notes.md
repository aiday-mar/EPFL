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
