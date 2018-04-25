#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

#include "common.h"
#include "handover1.h"
#include "handover2.h"

int main(int argc, char **argv)
{
	if (argc != 4)
	{
		printf("Usage: %s 1 2 logfile.pre\n", argv[0]);
		printf("1: number of runs\n");
		printf("2: number of threads (for multi-threaded version)\n");
		return 0;
	}
	
	n_runs = atoi(argv[1]);
	n_threads = atoi(argv[2]);

	read_log(argv[argc-1]);
	
	n_matrix = n_users*n_users;
	
	int* matrix1 = handover1();
//	print_matrix(matrix1, n_matrix);

	int* matrix2 = handover2();
//	print_matrix(matrix2, n_matrix);

	printf("Check matrix: ");
	if (memcmp(matrix1, matrix2, n_matrix*sizeof(int)) == 0)
	{
		printf("OK");
	}
	else
	{
		printf("Not OK");
	}
	printf("\n");

	free(matrix1);
	free(matrix2);

	free_log();

	return 0;
}
