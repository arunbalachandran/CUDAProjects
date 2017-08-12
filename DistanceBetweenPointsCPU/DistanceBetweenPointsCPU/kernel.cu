// to avoid highlight problems
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h> // imported for rand() which generates a number between 0 & RAND_MAX
#include <time.h>   // imported for the time() function and also the clock function
#include <limits>	// for a large value
#include <cmath>    // for exponentiation
using namespace std;

void FindClosestPoint(float3 *points, const int numberPoints, int *closestPoint)
{
	// now find the closest point to each point
	// 'i' represents the current point that we are finding the closest point to!	
	for (int i = 0; i < numberPoints; i++)
	{
		int distanceBetweenPoints = numeric_limits<int>::max(), tempDistance=0;
		for (int j = 0; j < numberPoints; j++)
			if (i != j)		// dont check the distance between the point and itself
			{
				tempDistance = pow((points[i].x - points[j].x), 2) + pow((points[i].y - points[j].y), 2);
				if (tempDistance < distanceBetweenPoints)
				{
					distanceBetweenPoints = tempDistance;
					closestPoint[i] = j;
				}
			}
	}
	// display the closest points
	cout << "The closest points :" << endl;
	for (int i = 0; i < numberPoints; i++)
		cout << i << "\t" << closestPoint[i] << endl;
}

int main()
{
	srand(time(NULL));  // used to initialize the seed for the random number generator
	const int numberPoints = 1000;
	clock_t startTime, endTime;
	float3 *points = new float3[numberPoints];
	int *closestPoint = new int[numberPoints];
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
	// now find the distance between all points
	startTime = clock();
	FindClosestPoint(points, numberPoints, closestPoint);
	endTime = clock() - startTime;
	cout << "Time it took was " << ((float) endTime / CLOCKS_PER_SEC) << endl;
}
