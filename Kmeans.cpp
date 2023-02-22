#include "Kmeans.h"
#include <cmath>
#include <omp.h>

using namespace std;


Kmeans::Kmeans(int dim,int numPoints,int numClusters,float* points, float* centroids,int* membersCounter)
{
    this->dim=dim;
    this->numPoints=numPoints;
    this->numClusters=numClusters;
    this->points=points;
    this->centroids=centroids;
    this->membersCounter=membersCounter;
   
  
}

void Kmeans::kmeansIteration()
{
    float* newCentroids= new float[numClusters*dim]{0};
    
#pragma omp parallel for
    for(int n=0;n<numPoints;n++)
    {
       int nearestCluster=0;
       int minDistance=100000000;
       
       //determina il centroide più vicino per ogni punto n 
       for(int c=0;c<numClusters;c++)
       {
        float euclideanDistance=0;
        for(int d=0;d<dim;d++)
        {
           euclideanDistance+=pow(*(points+n*dim+d)-*(centroids+c*dim+d),2);
        }
        euclideanDistance=sqrt(euclideanDistance);
        if(euclideanDistance<minDistance)
        {    
        minDistance=euclideanDistance;
        nearestCluster=c;
        }
       }
       
       //sommo i punti nel corrispondente centroide più vicino
       for(int j=0;j<dim;j++)
       {
         *(newCentroids+nearestCluster*dim+j)+=*(points+n*dim+j);
       }
       //mantiene il numero di punti in ciascun cluster
       membersCounter[nearestCluster]++;      
    }

    //calcola i nuovi centroidi
#pragma omp parallel for collapse(2)
    for(int i=0;i<numClusters;i++)
    {
        for(int j=0;j<dim;j++)
        {
            *(newCentroids+i*dim+j)=(*(newCentroids+i*dim+j))/membersCounter[i];
        }
    }
    
    //aggiorna i centroidi con i nuovi valori
#pragma omp parallel for collapse(2)
    for(int i=0;i<numClusters;i++)
    {
        for(int j=0;j<dim;j++)
        {
            *(centroids+i*dim+j)=*(newCentroids+i*dim+j);
        }
    }
}


void Kmeans::computeKmeans(int MAX_ITER)
{
    for(int i=0;i<MAX_ITER;i++)
    {
      for(int i=0;i<numClusters;i++){
        membersCounter[i]=0;
      }
      kmeansIteration();
    }
}