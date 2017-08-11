#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
// the CUDA runtime is needed for __global__
using namespace std;

// kernel that the host will execute on the GPU
__global__ void AddIntegers(int *a, int *b)
{
	a[0] += b[0];
}

int main()
{
	int a = 7, b = 6;
	int *da, *db;
	// allocate memory for the device pointers
	cudaMalloc(&da, sizeof(int));
	cudaMalloc(&db, sizeof(int));
	// copy data from host to device
	cudaMemcpy(da, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(db, &b, sizeof(int), cudaMemcpyHostToDevice);
	// actual function call
	AddIntegers << <1, 1 >> >(da, db);
	// copy answer back to Host
	cudaMemcpy(&a, da, sizeof(int), cudaMemcpyDeviceToHost);
	cout << "The answer is " << a << endl;
	// free allocated memory on device
	cudaFree(da);
	cudaFree(db);
	return 0;
}