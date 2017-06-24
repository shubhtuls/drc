#ifndef _INCL_MLE_DEPTH_SEM
#define _INCL_MLE_DEPTH_SEM

#include "stdlib.h"
#include <math.h>
#include "grid.h"
#include <string.h>

typedef struct mleDepthSem{
    grid* g;
    double maxDepth;
    long int* sz;
    int useProj;
} mleDepthSem;

mleDepthSem* mleDepthSem_init(grid* g, double maxDepth, int useProj);

void mleDepthSem_forward(mleDepthSem* rp, double *predsGeom, int* predsSem, double *origins, double* directions, double* inferredDepth, int* inferredClass, int bs, int nrays);

#endif
