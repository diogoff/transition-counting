#define CUDA_CHECK(call)										\
{																\
	cudaError_t error = call;									\
	if (error != cudaSuccess)									\
	{															\
        printf("CUDA Error: %s\n", cudaGetErrorString(error));	\
		exit(1);												\
	}															\
}

inline double seconds()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec * 1.e-6;
}

typedef struct Event
{
	int caseid;
	int task;
	int user;
} Event;

typedef struct Pos
{
	int begin;
	int end;
} Pos;

int n_events;
Event* all_events;

int n_caseids;
int caseid_len;
char* caseid_values;

int n_tasks;
int task_len;
char* task_values;

int n_users;
int user_len;
char* user_values;

Pos* pos_caseid;

void read_log(char fname[])
{
	printf("Reading: %s\n", fname);
	FILE* fin = fopen(fname, "r");
	if (fin == NULL)
	{
		printf("Could not open file.\n");
		exit(1);
	}
	
	fread(&n_events, sizeof(int), 1, fin);
//	printf("Events: %d\n", n_events);
	all_events = (Event*)malloc(n_events*sizeof(Event));
	fread(all_events, sizeof(Event), n_events, fin);
//	for (int i=0; i<n_events; i++) printf("%d %d %d\n", all_events[i].caseid, all_events[i].task, all_events[i].user);
	
	fread(&n_caseids, sizeof(int), 1, fin);
//	printf("Cases: %d\n", n_caseids);
	fread(&caseid_len, sizeof(int), 1, fin);
	caseid_values = (char*)malloc(n_caseids*caseid_len*sizeof(char));
	fread(caseid_values, sizeof(char), n_caseids*caseid_len, fin);
//	printf("caseid_values:");
//	for (int i=0; i<n_caseids; i++) printf(" %s", caseid_values+i*caseid_len);
//	printf("\n");
	
	fread(&n_tasks, sizeof(int), 1, fin);
//	printf("Tasks: %d\n", n_tasks);
	fread(&task_len, sizeof(int), 1, fin);
	task_values = (char*)malloc(n_tasks*task_len*sizeof(char));
	fread(task_values, sizeof(char), n_tasks*task_len, fin);
//	printf("task_values:");
//	for (int i=0; i<n_tasks; i++) printf(" %s", task_values+i*task_len);
//	printf("\n");
	
	fread(&n_users, sizeof(int), 1, fin);
//	printf("Users: %d\n", n_users);
	fread(&user_len, sizeof(int), 1, fin);
	user_values = (char*)malloc(n_users*user_len*sizeof(char));
	fread(user_values, sizeof(char), n_users*user_len, fin);
//	printf("user_values:");
//	for (int i=0; i<n_users; i++) printf(" %s", user_values+i*user_len);
//	printf("\n");
	
	pos_caseid = (Pos*)malloc(n_caseids*sizeof(Pos));
	fread(pos_caseid, sizeof(Pos), n_caseids, fin);
//	printf("pos_caseid:");
//	for (int i=0; i<n_caseids; i++) printf("%d %d\n", pos_caseid[i].begin, pos_caseid[i].end);

	fclose(fin);
}

void free_log()
{
	free(all_events);
	free(caseid_values);
	free(task_values);
	free(user_values);
	free(pos_caseid);
}

void print_matrix(int* matrix, int n_matrix)
{
	printf("Matrix:");
	for (int i=0; i<n_matrix; i++)
	{
		printf(" %d", matrix[i]);
	}
	printf("\n");
}
