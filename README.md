# Parallelization of Transition Counting for Process Mining on Multi-core CPUs and GPUs

This repository contains the source code for the paper [Parallelization of Transition Counting for Process Mining on Multi-core CPUs and GPUs](http://web.tecnico.ulisboa.pt/diogo.ferreira/papers/ferreira17parallelization.pdf) presented at the _12th International Workshop on Business Process Intelligence_ (BPI 2016) in Rio de Janeiro, Brazil, September 2016.

### Source files

The source code comprises the following files:

- `simulator.py` - A Python script to generate event logs of various sizes. The input parameter for this script is the number of cases that the event log should contain.

- `preproc.py` - A Python script to do the preprocessing of an event log. This script receives a CSV file (.csv) as input and produces a binary file (.pre) as output.

- `common.h` - Some common routines and variables used by all programs, namely to read the event log in the binary format (.pre) created by the previous script.

- `flow1.h` - The single-threaded CPU version of the flow algorithm.

- `flow2.h` - The multi-threaded CPU version of the flow algorithm.

- `flow3.h` - The GPU version of the flow algorithm.

- `flow_cpu.cc` - The C/C++ program that runs the multi-threaded CPU version of the flow algorithm.

- `flow_gpu.cu` - The CUDA C program that runs the GPU version of the flow algorithm.

- `handover1.h` - The single-threaded CPU version of the handover algorithm.

- `handover2.h` - The multi-threaded CPU version of the handover algorithm.

- `handover3.h` - The GPU version of the handover algorithm.

- `handover_cpu.cc` - The C/C++ program that runs the multi-threaded CPU version of the handover algorithm.

- `handover_gpu.cu` - The CUDA C program that runs the GPU version of the handover algorithm.

- `together1.h` - The single-threaded CPU version of the together algorithm.

- `together2.h` - The multi-threaded CPU version of the together algorithm.

- `together3.h` - The GPU version of the together algorithm.

- `together_cpu.cc` - The C/C++ program that runs the multi-threaded CPU version of the together algorithm.

- `together_gpu.cu` - The CUDA C program that runs the GPU version of the together algorithm.

- `runtests.py` - A Python script that runs a series of tests and generates results in a similar form to Table 2 in the paper.

### How to cite this work

See the [publisher's website](https://link.springer.com/chapter/10.1007%2F978-3-319-58457-7_3) to download a citation in the desired format.
