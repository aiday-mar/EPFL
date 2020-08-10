# Parallel and High Performance Computing

The Von Neumann architecture consists in an ALU (an arithmetical and logical unit) along with a control unit in a central processing unit together. There is a memory unit linked to the ALU, as well as input and output devices. The system bus consists in the control, the address and the data bus. The address bus gives us instructions on where to find the data, the data bus gives instructions on what to find, the control bus gives instructions on what to do with the data (for example : read, write, fetch, etc...). The CPU performance is measured in FLOPS. The memory bandwidth in bytes per second and the memory latency in seconds. The peak values are calculated as follows :

Peak CPU performance = CPU Frequency X Number of Operations per clock cycle X size of the largest vector X number of cores
Memory bandwidth = RAM Frequency X Number of transfers per clock cycle X Bus width X number of interfaces
Memory latency depends on the size of the data. Usualy given by the constructor

There are limits to how fast a CPU can run and its circuitry cannot always keep up with an overclocked speed. If the clock tells the CPU to execute instructions too quickly, the processing will not be completed before the next instruction is carried out. If the CPU cannot keep up with the pace of the clock, the data is corrupted. CPUs can also overheat if they are forced to work faster than they were designed to work. (https://www.bbc.co.uk/bitesize/guides/zmb9mp3/revision/2)

Memory bandwidth and latency are key considerations in almost all applications, but especially so for GPU applications. Bandwidth refers to the amount of data that can be moved to or from a given destination. In the GPU case we’re concerned primarily about the global memory bandwidth. Latency refers to the time the operation takes to complete.

The High Performance Linpack seems to be a test where an n x n system of linear equations is solved using Gauss pivoting.

