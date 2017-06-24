#include<stdlib.h>
#include "grid.h"
#include "math.h"

void addMult(double *p1, double *p2, double c, int Nd){
    for(int d=0;d<Nd;d++) p1[d] += p2[d]*c;
    return;
}

double distFunc(double *p1, double *p2, int Nd){
    double sqDist=0;
    double diff;
    for(int d=0;d<Nd;d++){diff = p1[d]-p2[d]; sqDist += diff*diff;}
    return sqrt(sqDist);
}

void copyArrInt(long int *to, long int *from, int Nd){
    for(int d=0;d<Nd;d++) to[d] = from[d];
}

void copyArr(double *to, double *from, int Nd){
    for(int d=0;d<Nd;d++) to[d] = from[d];
}

grid* grid_init(int nDim, double* minBounds, double* maxBounds, double* focal){
    grid* g;
    if((g = malloc(sizeof *g)) != NULL){
        g->nDim = nDim;
        g->minBounds = minBounds;
        g->maxBounds = maxBounds;
        g->focal = focal;
    }
    return g;
}
