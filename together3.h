// ---------------------------------------------------------------------------------------
// GPU version

int n_threads_per_block;

__global__ void kernel_participants(Event* d_all_events, int n_events, bool* d_participants, int n_users)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_events)
	{
		int caseid = d_all_events[i].caseid;
		int user = d_all_events[i].user;
		d_participants[caseid*n_users+user] = true;
	}
}

__global__ void kernel_pairs(bool* d_participants, int n_transitions, int2* d_pairs, int n_pairs, int* d_transitions, int n_users)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_transitions)
	{
		int caseid = i / n_pairs;
		int k = i - caseid*n_pairs;
		int user0 = d_pairs[k].x;
		int user1 = d_pairs[k].y;
		if (d_participants[caseid*n_users+user0] && d_participants[caseid*n_users+user1])
		{
			d_transitions[i] = n_users*user0 + user1 + 1;
		}
	}
}

__global__ void kernel_matrix(int* d_results, int n_matrix, int* d_matrix)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_matrix)
	{
		if (d_results[i] > 0)
		{
			d_matrix[d_results[i]-1] = d_results[n_matrix+i];
		}
	}
}

int* together3()
{
	printf("Running GPU version (%d run%s, %d thread%s per block)\n", n_runs, (n_runs > 1) ? "s" : "", n_threads_per_block, (n_threads_per_block > 1) ? "s" : "");

	int* matrix = (int*)malloc(n_matrix*sizeof(int));
	
	try
	{
		Event* d_all_events;
		CUDA_CHECK(cudaMalloc(&d_all_events, n_events*sizeof(Event)));
		CUDA_CHECK(cudaMemcpy(d_all_events, all_events, n_events*sizeof(Event), cudaMemcpyHostToDevice));
		
		int n_participants = n_caseids*n_users;
		bool* d_participants;
		CUDA_CHECK(cudaMalloc(&d_participants, n_participants*sizeof(bool)));

		int threads0 = n_threads_per_block;
		int blocks0 = (int)ceil((double)n_events/(double)threads0);
		while ((blocks0 > 65535) && (threads0 + n_threads_per_block <= 1024))
		{
			threads0 += n_threads_per_block;
			blocks0 = (int)ceil((double)n_events/(double)threads0);
		}

		int n_pairs = n_users*(n_users-1)/2;
		int2* pairs = (int2*)malloc(n_pairs*sizeof(int2));
		int k = 0;
		for(int u0=0; u0<n_users-1; u0++)
		{
			for(int u1=u0+1; u1<n_users; u1++)
			{
				pairs[k].x = u0;
				pairs[k].y = u1;
				k++;
			}
		}
		int2* d_pairs;
		CUDA_CHECK(cudaMalloc(&d_pairs, n_pairs*sizeof(int2)));
		CUDA_CHECK(cudaMemcpy(d_pairs, pairs, n_pairs*sizeof(int2), cudaMemcpyHostToDevice));
		free(pairs);
		
		int n_transitions = n_caseids*n_pairs;
		thrust::device_vector<int> dv_transitions(n_transitions);
		int* d_transitions = thrust::raw_pointer_cast(&dv_transitions[0]);

		int threads1 = n_threads_per_block;
		int blocks1 = (int)ceil((double)n_transitions/(double)threads1);
		while ((blocks1 > 65535) && (threads1 + n_threads_per_block <= 1024))
		{
			threads1 += n_threads_per_block;
			blocks1 = (int)ceil((double)n_transitions/(double)threads1);
		}

		thrust::constant_iterator<int> const_iter(1);

		int n_results = 2*n_matrix;
		thrust::device_vector<int> dv_results(n_results);
		int* d_results = thrust::raw_pointer_cast(&dv_results[0]);

		int threads2 = n_threads_per_block;
		int blocks2 = (int)ceil((double)n_matrix/(double)threads2);
		while ((blocks2 > 65535) && (threads2 + n_threads_per_block <= 1024))
		{
			threads2 += n_threads_per_block;
			blocks2 = (int)ceil((double)n_matrix/(double)threads2);
		}

		int* d_matrix;
		CUDA_CHECK(cudaMalloc(&d_matrix, n_matrix*sizeof(int)));

		double total = 0.0;
		for(int r=0; r<n_runs; r++)
		{
			// -------------------------------------------------------------
			double t0 = seconds();

			CUDA_CHECK(cudaMemset(d_participants, 0, n_participants*sizeof(bool)));
			CUDA_CHECK(cudaMemset(d_transitions, 0, n_transitions*sizeof(int)));
			CUDA_CHECK(cudaMemset(d_results, 0, n_results*sizeof(int)));
			CUDA_CHECK(cudaMemset(d_matrix, 0, n_matrix*sizeof(int)));

			kernel_participants<<<blocks0, threads0>>>(d_all_events, n_events, d_participants, n_users);

			kernel_pairs<<<blocks1, threads1>>>(d_participants, n_transitions, d_pairs, n_pairs, d_transitions, n_users);
			
			thrust::sort(dv_transitions.begin(), dv_transitions.end());
			
			thrust::reduce_by_key(dv_transitions.begin(), dv_transitions.end(), const_iter, dv_results.begin(), dv_results.begin() + n_matrix);
			
			kernel_matrix<<<blocks2, threads2>>>(d_results, n_matrix, d_matrix);

			double t1 = seconds();
			// -------------------------------------------------------------
			total += (t1-t0);
		}
		printf("Time: %f\n", total/n_runs);

		CUDA_CHECK(cudaMemcpy(matrix, d_matrix, n_matrix*sizeof(int), cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaFree(d_all_events));
		CUDA_CHECK(cudaFree(d_participants));
		CUDA_CHECK(cudaFree(d_matrix));
		CUDA_CHECK(cudaFree(d_pairs));
	}
	catch(thrust::system_error& error)
	{
		printf("Thrust Error: %s\n", error.what());
		exit(1);
	}
	catch(std::bad_alloc& error)
	{
		printf("Thrust Error: out of memory\n");
		exit(1);
	}

	return matrix;
}
