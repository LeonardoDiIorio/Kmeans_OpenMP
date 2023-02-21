#include "Kmeans.h"

//metodi classe Point

Point::Point(int dimensions, int pointId)
{  
    this->pointId=pointId;
    this->dimensions=dimensions;
    this->components.resize(dimensions);
}


double Point::getComponent(int index)
{
   return this->components[index];
}

void Point::setComponent(double value,int index)
{
    this->components[index]=value;
    return;
}




//metodi classe Cluster

Cluster::Cluster(vector<double> firstCentroid)
{
    this->centroid=firstCentroid;
    return;
}

void Cluster::addPoint(Point newMember)
{
    this->members.push_back(newMember);
    return;
}

void Cluster::clearCluster()
{
    this->members.clear();
}

double Cluster::getCentroidComponent(int index)
{
  return this->centroid[index];
}

vector<Point> Cluster::getMembers()
{
  return this->members;
}

void Cluster::setCentroidComponent(int index,double value)
{
  this->centroid[index]=value;
}



//metodi classe Kmeans

Kmeans::Kmeans(int numberOfPoints,int dimensions,  int numberOfClusters)
{
    this->numberOfPoints=numberOfPoints;
    this->dimensions=dimensions;
    this->numberOfClusters=numberOfClusters;

}


void Kmeans::generatePoints(double min, double max)
{
    
    //costruisco tutti i punti
    for(int n=0;n<numberOfPoints;n++)
    {
      inputPoints.emplace_back(dimensions,n);
    }
 //   cout<<inputPoints.size()<<"\n";
       
   
    
    uniform_real_distribution<double> unif(min,max);                                          
    default_random_engine re;
    
    //inizializzo tutti i punti
    #pragma omp parallel for collapse(2)
    for(int i=0;i<numberOfPoints;i++)
    {
      for(int j=0;j<dimensions;j++)
      {
        inputPoints[i].setComponent(unif(re),j);
      }
    }     
}


void  Kmeans::initializeCentroids()
{   
    
    #pragma omp parallel for 
    for(int i=0;i<numberOfClusters;i++)
    {
      int randomPoint= rand()%numberOfPoints;
      vector<double> centroidComponents;
      for(int j=0;j<dimensions;j++)
      {
        double value = inputPoints[randomPoint].getComponent(j);
        centroidComponents.push_back(value);
      }
      #pragma omp critical
      clusters.emplace_back(centroidComponents);
    }
}


void Kmeans::assignPoints()
{
   
   #pragma omp parallel for //vedere come gestire i for annidati
   for(int i=0;i<numberOfPoints;i++)
   {
    double minDistance=100000000;
    int nearestClusterIndex= numberOfClusters+1;
    for(int j=0;j<numberOfClusters;j++)
    {
      double euclideaDistance=0;
      for(int d=0;d<dimensions;d++)
      {
         euclideaDistance+=pow(inputPoints[i].getComponent(d)-clusters[j].getCentroidComponent(d),2);
      }
      euclideaDistance=sqrt(euclideaDistance);
      if(euclideaDistance<minDistance)
      {
       
        minDistance=euclideaDistance;
        nearestClusterIndex=j;
      }
    }
    #pragma omp critical
    clusters[nearestClusterIndex].addPoint(inputPoints[i]);
   }
}


void Kmeans::updateCentroids()
{
   #pragma omp parallel for 
    for(int i=0;i<numberOfClusters;i++)
    {
      vector<Point> members = clusters[i].getMembers();
      for(int d=0;d<dimensions;d++)
      {
        double updatedValue=0;
        for(int j=0;j<members.size();j++)
        {
          updatedValue+=members[j].getComponent(d);
        }
        updatedValue=(updatedValue/members.size());
        clusters[i].setCentroidComponent(d,updatedValue);
      }
    }
}


void Kmeans::computeError()
{
  double SSE=0;
  double interClusterSum;
    #pragma omp parallel for 
    for(int i=0;i<numberOfClusters;i++)
    {
      vector<Point> members = clusters[i].getMembers();
      double interClusterSum=0;
      for(int j=0;j<members.size();j++)
      {
        
        double squareDistance=0;
        for(int d=0;d<dimensions;d++)
        {        
          squareDistance+=pow(members[j].getComponent(d)-clusters[i].getCentroidComponent(d),2);
        }
        interClusterSum+=sqrt(squareDistance);
      } 
      SSE+=interClusterSum;
    }
 // cout<<"Errore:"<<SSE<<"\n";
}

void Kmeans::computeKmeans(int numberOfIterations)
{

  this->numberOfIterations=numberOfIterations;
  generatePoints(0,80); //intervallo di generazione dei valori dei punti
  initializeCentroids();
  for(int i=0;i<numberOfIterations;i++)
  {
    assignPoints();
    updateCentroids();
    computeError();
    int sum=0;
    #pragma omp parallel for
    for(int j=0;j<numberOfClusters;j++)
    {
      sum+=clusters[j].getMembers().size();
      clusters[j].clearCluster();
    }
  //  cout<<sum<<"\n";
  }
}





