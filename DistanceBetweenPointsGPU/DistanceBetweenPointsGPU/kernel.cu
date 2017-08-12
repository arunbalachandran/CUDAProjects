// to avoid highlight problems
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h> // imported for rand() which generates a number between 0 & RAND_MAX
#include <time.h>   // imported for the time() function and also the clock function
#include <limits>	// for a large value
#include <cmath>    // for exponentiation
using namespace std;

__global__ void FindClosestPoint(float3 *points, int *closestPoint, const int numberPoints)
{
	// used to identify the thread that is currently running
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// now find the closest point to each point
	// 'i' represents the current point that we are finding the closest point to!	
	int distanceBetweenPoints = 9999999, tempDistance = 0;
	for (int j = 0; j < numberPoints; j++)
		if (idx != j)		// dont check the distance between the point and itself
		{
			tempDistance = pow((points[idx].x - points[j].x), 2) + pow((points[idx].y - points[j].y), 2);
			if (tempDistance < distanceBetweenPoints)
			{
				distanceBetweenPoints = tempDistance;
				closestPoint[idx] = j;
			}
		}
}

int main()
{
	srand(time(NULL));  // used to initialize the seed for the random number generator
	const int numberPoints = 1000;
	clock_t startTime, endTime;
	float3 *points = new float3[numberPoints];
	float3 *pointsDeviceCopy;
	int *closestPointDevice, *closestPoint = new int[numberPoints];
	// initialize the points with random numbers	
	for (int i = 0; i < numberPoints; i++)
	{
		points[i].x = rand() % 1000;
		points[i].y = rand() % 1000;
		points[i].z = rand() % 1000;
	}
	// print the points initialized
	for (int i = 0; i < numberPoints; i++)
		cout << points[i].x << "\t" << points[i].y << "\t" << points[i].z << endl;
	cout << endl;
	
	// initialize memory in the GPU for calculation
	if (cudaMalloc(&pointsDeviceCopy, sizeof(float3) * numberPoints) != cudaSuccess)
	{
		cout << "Couldn't initialize memory in the GPU for pointsDeviceCopy" << endl;
		delete[] points;
		delete[] closestPoint;
		return 0;
	}

	if (cudaMalloc(&closestPointDevice, sizeof(int) * numberPoints) != cudaSuccess)
	{
		cout << "Couldn't initialize memory in the GPU for closestPointDevice" << endl;
		cudaFree(pointsDeviceCopy);
		delete[] points;
		delete[] closestPoint;
		return 0;
	}
	
	if (cudaMemcpy(pointsDeviceCopy, points, sizeof(float3) * numberPoints, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "Could not copy points to pointsDeviceCopy" << endl;
		cudaFree(pointsDeviceCopy);
		cudaFree(closestPointDevice);
		delete[] points;
		delete[] closestPoint;
		return 0;
	}
	
	// now find the distance between all points
	startTime = clock();
	// since a block can have upto 1024 elements, we can use a single block
	FindClosestPoint<<<1, numberPoints>>>(pointsDeviceCopy, closestPointDevice, numberPoints);
	if (cudaMemcpy(closestPoint, closestPointDevice, sizeof(int) * numberPoints, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cout << "Could not get the output!";
		cudaFree(pointsDeviceCopy);
		cudaFree(closestPointDevice);
		delete[] points;
		delete[] closestPoint;
		return 0;
	}

	endTime = clock() - startTime;
	delete[] points;
	delete[] closestPoint;
	cudaFree(closestPointDevice);
	cudaFree(pointsDeviceCopy);
	cout << "Time it took was " << ((float)endTime / CLOCKS_PER_SEC) << endl;
	return 0;
}
