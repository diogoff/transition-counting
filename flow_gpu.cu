#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include "common.h"
#include "flow1.h"
#include "flow3.h"

int main(int argc, char **argv)
{
	if (argc != 4)
	{
		printf("Usage: %s 1 2 logfile.pre\n", argv[0]);
		printf("1: number of runs\n");
		printf("2: number of threads per block (for GPU version)\n");
		return 0;
	}
	
	n_runs = atoi(argv[1]);
	n_threads_per_block = atoi(argv[2]);
	
	read_log(argv[argc-1]);
	
	n_matrix = n_tasks*n_tasks;

	int* matrix1 = flow1();
//	print_matrix(matrix1, n_matrix);

	int* matrix3 = flow3();
//	print_matrix(matrix3, n_matrix);

	printf("Check matrix: ");
	if (memcmp(matrix1, matrix3, n_matrix*sizeof(int)) == 0)
	{
		printf("OK");
	}
	else
	{
		printf("Not OK");
	}
	printf("\n");

	free(matrix1);
	free(matrix3);

	free_log();

	return 0;
}
