#include "Kmeans.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include<omp.h>

using namespace std;

int main()
{
  int maxThreads=12;
  int numTest=1;
 
  ofstream myfile;
  myfile.open ("kmeans.csv");
  for(int k=2;k<maxThreads+1;k+=2)
  {
    omp_set_num_threads(k);
    int time=0;
    Kmeans kmeansInstance(1000,10,280);
    auto start = chrono::system_clock::now();
    kmeansInstance.computeKmeans(40);
    auto end = chrono::system_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);
    time += elapsed.count();
    cout << "\t\tTime: " << time << endl;
    myfile << time<<"\n";  
  }
  myfile.close();
  return 0;
}