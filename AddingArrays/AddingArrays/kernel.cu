#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
#include <ctime>
#include <stdlib.h>
// imported for the random functionality
using namespace std;

__global__ void AddIntegers(int *arr1, int *arr2, int num_elements)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < num_elements)
	{
		arr1[id] += arr2[id];
	}
}

int main()
{
	srand(time(NULL));
	int num_elements = 100;
	int *array1, *array2;
	array1 = new int[num_elements];
	array2 = new int[num_elements];
	// initialize with random numbers	
	for (int i = 0; i < num_elements; i++)
	{
		array1[i] = rand() % 1000;
		array2[i] = rand() % 1000;
	}
	// print the numbers that were initialized
	cout << "The numbers that were initialized were : " << endl;
	for (int i = 0; i < num_elements; i++)
		cout << array1[i] << " " << array2[i] << endl;
	cout << endl;

	// now initialize kernel
	int *deviceArray1, *deviceArray2;
	if (cudaMalloc(&deviceArray1, sizeof(int) * num_elements) != cudaSuccess)
	{
		cout << "Couldn't initialize deviceArray1!" << endl;
		return 0;
	}
	if (cudaMalloc(&deviceArray2, sizeof(int) * num_elements) != cudaSuccess)
	{
		cout << "Couldn't initialize deviceArray2!" << endl;
		cudaFree(deviceArray1);
		return 0;
	}
	// now copy the data from the local memory to GPU memory
	if (cudaMemcpy(deviceArray1, array1, sizeof(int) * num_elements, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "Could not copy to deviceArray1!" << endl;
		cudaFree(deviceArray1);
		cudaFree(deviceArray2);
		return 0;
	}
	if (cudaMemcpy(deviceArray2, array2, sizeof(int) * num_elements, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "Could not copy to deviceArray2!" << endl;
		cudaFree(deviceArray1);
		cudaFree(deviceArray2);
		return 0;
	}
	
	AddIntegers<<<num_elements / 256 + 1, 256 >>>(deviceArray1, deviceArray2, num_elements);
	// now copy the data back from the GPU memory to local memory
	if (cudaMemcpy(array1, deviceArray1, sizeof(int) * num_elements, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		delete[] array1;
		delete[] array2;
		cudaFree(deviceArray1);
		cudaFree(deviceArray2);
		return 0;
	}

	for (int i = 0; i < num_elements; i++)
		cout << array1[i] << endl;

	cudaFree(deviceArray1);
	cudaFree(deviceArray2);
	delete[] array1;
	delete[] array2;
	return 0;
}