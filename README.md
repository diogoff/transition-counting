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

### How to compile the code

Operating system: Ubuntu (or another Linux distro).

Check that you have all the necessary pre-requisites, namely [CUDA](https://developer.nvidia.com/cuda-downloads). Also, make sure that CUDA is properly configured, as described [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions). A good place to configure the environment variables (`PATH` and `LD_LIBRARY_PATH`) is in `~/.bashrc`.

Use the `Makefile` to compile the code. However, before running `make`, change the value of the `ARCH` variable to the compute capability of your GPU. For a list of GPUs and their compute capabilities, check [this page](https://developer.nvidia.com/cuda-gpus).

### Generating the event logs

Use `simulator.py` to generate event logs with size ranging from 10 to 10<sup>7</sup> cases. For this purpose, run the following commands:

- `python simulator.py 10 > eventlog_10.csv`
- `python simulator.py 100 > eventlog_100.csv`
- `python simulator.py 1000 > eventlog_1000.csv`
- `python simulator.py 10000 > eventlog_10000.csv`
- `python simulator.py 100000 > eventlog_100000.csv`
- `python simulator.py 1000000 > eventlog_1000000.csv`
- `python simulator.py 10000000 > eventlog_10000000.csv`

Note that generating the larger event logs may take a significant amount of time (several minutes).

### Preprocessing the event logs

After generating the event logs, run the following commands to preprocess them:

- `python preproc.py 0 1 2 3 eventlog_10.csv`
- `python preproc.py 0 1 2 3 eventlog_100.csv`
- `python preproc.py 0 1 2 3 eventlog_1000.csv`
- `python preproc.py 0 1 2 3 eventlog_10000.csv`
- `python preproc.py 0 1 2 3 eventlog_100000.csv`
- `python preproc.py 0 1 2 3 eventlog_1000000.csv`
- `python preproc.py 0 1 2 3 eventlog_10000000.csv`

The arguments `0 1 2 3` specify that the _case id_ is in column 0 (first column), _task_ is in column 1, _user_ is in column 2, and _timestamp_ is in column 3, respectively.

You may be surprised that preprocessing the largest event log can take about 5min. Most of this time is spent on reading the data (i.e. parsing the CSV file). This is precisely the reason why we decided to separate the preprocessing from the main programs. This way we can preprocess the event log only once, and then run several different programs on it, without having to reprocess it again.

### Running the programs

To run the multi-threaded CPU version of each algorithm, you can use instructions similar to the following:

- `./flow_cpu 100 4 eventlog_1000000.pre`
- `./handover_cpu 100 4 eventlog_1000000.pre`
- `./together_cpu 100 4 eventlog_1000000.pre`



### How to cite this work

See the [publisher's website](https://link.springer.com/chapter/10.1007%2F978-3-319-58457-7_3) to download a citation in the desired format.
