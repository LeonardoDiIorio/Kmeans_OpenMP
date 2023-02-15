
#ifndef KMEANS_H
#define KMEANS_H

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include<omp.h>


using namespace std;


class Point
{
private:
int pointId;
int dimensions;
vector<double> components;


public:
Point(int dimensions, int pointId);
double getComponent(int n);

void setComponent(double value,int index);


};



class Cluster
{
private:
vector<Point> members;
vector<double> centroid;

public:
Cluster(vector<double> firstCentroid);
void addPoint(Point newMember);
double getCentroidComponent(int index);
void setCentroidComponent(int index, double value);
void clearCluster();
vector<Point> getMembers();


};


class Kmeans
{
private:
vector<Point> inputPoints;
vector<Cluster> clusters;
int numberOfPoints;
int dimensions;
int numberOfClusters;
int numberOfIterations;


public:
Kmeans(int numberOfPoints,int dimensions, int numberOfClusters);
void generatePoints(double min,double max);
void initializeCentroids();
void assignPoints();
void updateCentroids();
void computeError();
void computeKmeans(int numberOfIterations);


};

#endif

