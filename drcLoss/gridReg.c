#include "stdlib.h"
#include <math.h>
#include "gridReg.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>

long int* gridReg_size(gridReg* g){
    long int* sz;
    int Nd = g->nDim;
    sz = malloc(Nd * sizeof(long int));
    for(int d=0;d<Nd;d++){
        sz[d] = (int)((g->maxBounds[d] - g->minBounds[d])/g->focal[d]);
    }
    return sz;
}

long int gridReg_nCells(gridReg *g){
    long int *sz = gridReg_size(g);
    long int nCells = 1;
    for(int d=0;d<g->nDim;d++){nCells *= sz[d];}
    free(sz);
    return nCells;
}

int gridReg_cellIndex(gridReg *g, double *point, long int *inds){
    int Nd = g->nDim;
    for(int d=0; d<Nd;d++){
        double pixel = point[d];
        if ((pixel >= g->maxBounds[d]) || (pixel < g->minBounds[d])){
            return 0;
        }
        inds[d]=(long) floor((pixel-g->minBounds[d])/g->focal[d]);
    }
    return 1;
}


int gridReg_initRayPoint(gridReg *g, double *origin, double *direction, double *initPoint, long int *initInds){
    int Nd = g->nDim;
    double t_min = inf;
    copyArr(initPoint, origin, Nd);
    addMult(initPoint, direction, eps, Nd);
    if(gridReg_cellIndex(g, initPoint, initInds) != 0){return 1;}

    double* candPoint = malloc(Nd*sizeof(double));
    double t, diff;

    for(int d=0; d<Nd;d++){
        if(direction[d] == 0) continue;
        double boundaryVals[2] = {g->minBounds[d], g->maxBounds[d]};
        for(int ix=0;ix<2;ix++){
            double bVal = boundaryVals[ix];
            t = (bVal - initPoint[d])/direction[d];
            if(t>0 && t<t_min){
                copyArr(candPoint, initPoint, Nd);
                addMult(candPoint, direction, t+eps, Nd);
                if(gridReg_cellIndex(g,candPoint, initInds) != 0) t_min = t;
            }
        }
    }
    //printf("t_min = %f\n",t_min);
    if(t_min == inf) return 0;
    addMult(initPoint, direction, t_min+eps, Nd);
    free(candPoint);

    return 1;
}


int gridReg_nextRayPoint(gridReg *g, double *point, double *direction, double *nextPoint, long int* nextInds){
    double t_min = inf;
    double t, diff;
    int Nd = g->nDim;
    for(int d=0;d<Nd;d++){
        if(direction[d] == 0) continue;
        if(direction[d] > 0) diff = g->focal[d] - fmod(point[d]- g->minBounds[d], g->focal[d]);
        else diff = fmod(point[d] - g->minBounds[d], g->focal[d]);
        t = diff/fabs(direction[d]);
        //printf("diff = %f, d = %d, t = %f\n",diff,d,t);
        if(t < t_min) t_min = t;
    }
    //printf("t_min = %f",t_min);
    copyArr(nextPoint, point, Nd);
    addMult(nextPoint, direction, t_min+eps, Nd);
    return gridReg_cellIndex(g, nextPoint, nextInds);
}

int gridReg_traceRay(gridReg *g, double *origin, double *direction, long int *indSeq, double *depthSeq){
    int Nd = g->nDim;
    double* pointPrev = malloc(Nd*sizeof(double));
    double* point = malloc(Nd*sizeof(double));
    long int* cellInd = malloc(Nd*sizeof(long int));
    assert(pointPrev != NULL);
    assert(point != NULL);
    assert(cellInd != NULL);
    int nc = 0;
    double depth;
    int isInside = gridReg_initRayPoint(g, origin, direction, point, cellInd);
    while(isInside > 0){
        depth = distFunc(origin, point, Nd);
        depthSeq[nc] = depth;
        //printf("%f\n",depth);
        copyArrInt(indSeq+Nd*nc, cellInd, Nd);
        copyArr(pointPrev, point, Nd);
        nc+=1;
        isInside = gridReg_nextRayPoint(g, pointPrev, direction, point, cellInd);
    }
    free(point);free(pointPrev);free(cellInd);
    return nc;
}
