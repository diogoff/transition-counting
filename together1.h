// ---------------------------------------------------------------------------------------
// CPU single-threaded version

int n_runs;
int n_matrix;

int* together1()
{
	printf("Running CPU single-threaded version (%d run%s)\n", n_runs, (n_runs > 1) ? "s" : "");

	int* matrix = (int*)malloc(n_matrix*sizeof(int));
	bool* participants = (bool*)malloc(n_users*sizeof(bool));

	double total = 0.0;
	for(int r=0; r<n_runs; r++)
	{
		// -------------------------------------------------------------
		double t0 = seconds();

		memset(matrix, 0, n_matrix*sizeof(int));

		for (int i=0; i<n_caseids; i++)
		{
			memset(participants, 0, n_users*sizeof(bool));
			
			for (int j=pos_caseid[i].begin; j<pos_caseid[i].end; j++)
			{
				participants[all_events[j].user] = true;
			}
			
			for (int u0=0; u0<n_users; u0++)
			{
				if (participants[u0])
				{
					for (int u1=u0+1; u1<n_users; u1++)
					{
						if (participants[u1])
						{
							matrix[n_users*u0+u1]++;
						}
					}
				}
			}
		}

		double t1 = seconds();
		// -------------------------------------------------------------
		total += (t1-t0);
	}
	printf("Time: %f\n", total/n_runs);
	
	free(participants);

	return matrix;
}
