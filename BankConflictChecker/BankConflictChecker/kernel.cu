#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>
using namespace std;

__global__ void BankConfKernel(unsigned long long *time)
{
	__shared__ float shared[1024];
	unsigned long long startTime = clock();
	// shared[0]++;  // a race condition, but neverthelesss a broadcast
	// shared[threadIdx.x*2]
	shared[threadIdx.x*32]++;
	unsigned long long finishTime = clock();
	*time = (finishTime - startTime);
}

int main()
{
	// do we need unsigned long long to represent the clock ticks?
	unsigned long long time;
	unsigned long long *timeDevice;
	cudaMalloc(&timeDevice, sizeof(unsigned long long));
	// there is an overhead for calling clock which should be subtracted from the time being displayed
	// number of runs
	for (int i = 0; i < 20; i++)
	{
		BankConfKernel<<<1, 32>>>(timeDevice);
		cudaMemcpy(&time, timeDevice, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		cout << "Time is " << (time) / 32 << endl;
	}
	cudaFree(timeDevice);
	cudaDeviceReset();  // used if you want to use the profiler
	return 0;
}