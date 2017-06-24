#include "stdlib.h"
#include <math.h>
#include "gridProj.h"
#include <string.h>
#include <stdio.h>

long int* gridProj_size(gridProj* g){
    long int* sz;
    int Nd = g->nDim;
    sz = malloc(Nd * sizeof(long int));
    for(int d=0;d<Nd-1;d++){
        sz[d] = (int)(g->maxBounds[d] - g->minBounds[d]);
    }
    sz[Nd-1] = (int)((log(g->maxBounds[Nd-1]) - log(g->minBounds[Nd-1]))/g->focal[Nd-1]);
    return sz;
}

long int gridProj_nCells(gridProj *g){
    long int *sz = gridProj_size(g);
    long int nCells = 1;
    for(int d=0;d<g->nDim;d++){nCells *= sz[d];}
    free(sz);
    return nCells;
}

int gridProj_cellIndex(gridProj *g, double *point, long int *inds){
    int Nd = g->nDim;
    double zVal = point[Nd-1];
    for(int d=0; d<Nd-1;d++){
        double pixel = g->focal[d]*point[d]/zVal;
        if ((pixel >= g->maxBounds[d]) || (pixel < g->minBounds[d])){
            return 0;
        }
        inds[d]=(long) floor(pixel-g->minBounds[d]);
    }
    if((zVal >= g->maxBounds[Nd-1]) || (zVal < g->minBounds[Nd-1])){return 0;}
    double z = log(zVal/g->minBounds[Nd-1])/g->focal[Nd-1];
    inds[Nd-1] = (long) floor(z);
    return 1;
}


int gridProj_initRayPoint(gridProj *g, double *origin, double *direction, double *initPoint, long int *initInds){
    int Nd = g->nDim;
    double t_min = inf;
    copyArr(initPoint, origin, Nd);
    addMult(initPoint, direction, eps, Nd);
    if(gridProj_cellIndex(g, initPoint, initInds) != 0){return 1;}

    double* candPoint = malloc(Nd*sizeof(double));

    for(int d=0; d<Nd;d++){
        if(direction[d] == 0) continue;
        double boundaryVals[2] = {g->minBounds[d], g->maxBounds[d]};
        for(int ix=0; ix < 2; ix++){
            double t;
            double bVal = boundaryVals[ix];
            if(d == Nd-1) t = (bVal - origin[d])/direction[d];
            else t = (bVal*initPoint[Nd-1] - g->focal[d]*initPoint[d])/(direction[d]*g->focal[d] - direction[Nd-1]*bVal);

            if(t>0 && t<t_min){
                copyArr(candPoint, initPoint, Nd);
                addMult(candPoint, direction, t+eps, Nd);
                if(gridProj_cellIndex(g,candPoint, initInds) != 0) t_min = t;
            }
        }
    }
    //printf("t_min = %f\n",t_min);
    if(t_min == inf) return 0;
    addMult(initPoint, direction, t_min+eps, Nd);
    free(candPoint);

    return 1;
}


int gridProj_nextRayPoint(gridProj *g, double *point, double *direction, double *nextPoint, long int* nextInds){
    double t_min = inf;
    int Nd = g->nDim;
    double dVal = point[Nd-1];
    double dirZ = direction[Nd-1];
    for(int d=0;d<Nd-1;d++){
        double fx = g->focal[d];
        double pixel = fx*point[d]/dVal;
        double p_f = floor(pixel);
        double p_c = ceil(pixel);
        double t1 = (p_f*dVal - fx*point[d])/(direction[d]*fx - dirZ*p_f);
        double t2 = (p_c*dVal - fx*point[d])/(direction[d]*fx - dirZ*p_c);
        if(t1 > 0 && t1 < t_min)t_min = t1;
        if(t2 > 0 && t2 < t_min)t_min = t2;
    }

    double z = log(dVal/g->minBounds[Nd-1])/g->focal[Nd-1];
    double zNext;
    if(dirZ > 0) zNext = ceil(z);
    else zNext = floor(z);

    double Z_next = exp(zNext*g->focal[Nd-1])*g->minBounds[Nd-1];
    double tZ = (Z_next - dVal)/dirZ;
    if(tZ > 0 && tZ < t_min)t_min = tZ;
    copyArr(nextPoint, point, Nd);
    addMult(nextPoint, direction, t_min+eps, Nd);
    return gridProj_cellIndex(g, nextPoint, nextInds);
}

int gridProj_traceRay(gridProj *g, double *origin, double *direction, long int *indSeq, double *depthSeq){
    int Nd = g->nDim;
    double* pointPrev = malloc(Nd*sizeof(double));
    double* point = malloc(Nd*sizeof(double));
    long int* cellInd = malloc(Nd*sizeof(long int));
    int nc = 0;
    double depth;
    int isInside = gridProj_initRayPoint(g, origin, direction, point, cellInd);
    while(isInside > 0){
        depth = distFunc(origin, point, Nd);
        depthSeq[nc] = depth;
        //printf("%f\n",depth);
        copyArrInt(indSeq+Nd*nc, cellInd, Nd);
        copyArr(pointPrev, point, Nd);
        nc+=1;
        isInside = gridProj_nextRayPoint(g, pointPrev, direction, point, cellInd);
    }
    free(point);free(pointPrev);free(cellInd);
    return nc;
}
