// ---------------------------------------------------------------------------------------
// CPU multi-threaded version

int n_threads;
pthread_t* threads;
int* local_matrix;
int* global_matrix;
pthread_mutex_t global_mutex;

void* worker_thread(void* arg)
{
	int i = (int)(long)arg;

	int* matrix = local_matrix + i*n_matrix;
	
	memset(matrix, 0, n_matrix*sizeof(int));
	
	while (i < n_caseids)
	{
		for (int j=pos_caseid[i].begin; j<pos_caseid[i].end-1; j++)
		{
			int task0 = all_events[j].task;
			int task1 = all_events[j+1].task;
			matrix[n_tasks*task0+task1]++;
		}

		i += n_threads;
	}

	// merge results into global matrix
	pthread_mutex_lock(&global_mutex);
	for (int i=0; i<n_matrix; i++)
	{
		global_matrix[i] += matrix[i];
	}
	pthread_mutex_unlock(&global_mutex);

	return NULL;
}

int* flow2()
{
	if (n_caseids < n_threads) n_threads = n_caseids;
	
	printf("Running CPU multi-threaded version (%d run%s, %d thread%s)\n", n_runs, (n_runs > 1) ? "s" : "", n_threads, (n_threads > 1) ? "s" : "");
	
	threads = (pthread_t*)malloc(n_threads*sizeof(pthread_t));
	local_matrix = (int*)malloc(n_threads*n_matrix*sizeof(int));
	global_matrix = (int*)malloc(n_matrix*sizeof(int));
	pthread_mutex_init(&global_mutex, NULL);

	double total = 0.0;
	for(int r=0; r<n_runs; r++)
	{
		// -------------------------------------------------------------
		double t0 = seconds();

		memset(global_matrix, 0, n_matrix*sizeof(int));

		for (int i=0; i<n_threads-1; i++)
		{
			pthread_create(&threads[i], NULL, worker_thread, (void*)(long)i);
		}
		
		worker_thread((void*)(long)(n_threads-1));

		for (int i=0; i<n_threads-1; i++)
		{
			pthread_join(threads[i], NULL);
		}

		double t1 = seconds();
		// -------------------------------------------------------------
		total += (t1-t0);
	}
	printf("Time: %f\n", total/n_runs);

	free(threads);
	free(local_matrix);
	pthread_mutex_destroy(&global_mutex);

	return global_matrix;
}
