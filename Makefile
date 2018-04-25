ARCH=sm_61

all : flow_cpu handover_cpu together_cpu flow_gpu handover_gpu together_gpu

flow_cpu : common.h flow1.h flow2.h flow_cpu.cc
	g++ -pthread flow_cpu.cc -o flow_cpu

handover_cpu : common.h handover1.h handover2.h handover_cpu.cc
	g++ -pthread handover_cpu.cc -o handover_cpu

together_cpu : common.h together1.h together2.h together_cpu.cc
	g++ -pthread together_cpu.cc -o together_cpu

flow_gpu : common.h flow1.h flow3.h flow_gpu.cu
	nvcc -arch=$(ARCH) flow_gpu.cu -o flow_gpu

handover_gpu : common.h handover1.h handover3.h handover_gpu.cu
	nvcc -arch=$(ARCH) handover_gpu.cu -o handover_gpu

together_gpu : common.h together1.h together3.h together_gpu.cu
	nvcc -arch=$(ARCH) together_gpu.cu -o together_gpu

clean:
	rm -f flow_cpu
	rm -f handover_cpu
	rm -f together_cpu
	rm -f flow_gpu
	rm -f handover_gpu
	rm -f together_gpu
