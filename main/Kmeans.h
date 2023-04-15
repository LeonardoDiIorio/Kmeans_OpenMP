#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>

using namespace std;

class Kmeans
{
    private:
    float* points;
    float* centroids;
    int* membersCounter;
    int dim;
    int numPoints;
    int numClusters;

    public:
    Kmeans(int dim,int numPoints,int numClusters,float* points, float* centroids, int* membersCounter);
    void kmeansIteration();
    void computeKmeans(int MAX_ITER);
};

#endif