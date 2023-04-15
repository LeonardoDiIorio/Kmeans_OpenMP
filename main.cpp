#include "Kmeans.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include<omp.h>

using namespace std;



int main()
{
  int maxThreads=16;
  int MAX_ITER=40;
  int dim=3;
  int numPoints=65536;
  int numClusters=256;
  float* points=new float[numPoints*dim]; //matrice dei punti
  float* centroids= new float[numClusters*dim]; //matrice dei centroidi
  int* membersCounter= new int[numClusters];


  uniform_real_distribution<float> unif(0,1000);                                          
  default_random_engine re;

  ofstream myfile;
  myfile.open ("kmeans.csv");
  
  for(int k=2;k<maxThreads+1;k+=2)
  {

    omp_set_num_threads(k);
        

    for(int i=0;i<numPoints;i++)
    {
        for(int j=0;j<dim;j++)
        {
          *(points+i*dim+j)=unif(re);
        }
    }
    

    for(int h=0;h<numClusters;h++)
    {
      for(int d=0;d<dim;d++)
      {
        *(centroids+h*dim+d)=*(points+h*dim+d);
      }
    }
    

    for(int c=0;c<numClusters;c++)
    {
      membersCounter[c]=0;
    }
    
    int time=0;
    Kmeans kmeansInstance(dim,numPoints,numClusters,points,centroids,membersCounter);
    auto start = chrono::system_clock::now();
    kmeansInstance.computeKmeans(MAX_ITER);
    auto end = chrono::system_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);
    time += elapsed.count();
    cout << "\t\tTime: " << time << endl;
    myfile << time<<"\n";  
  }
  myfile.close();
  return 0;
}