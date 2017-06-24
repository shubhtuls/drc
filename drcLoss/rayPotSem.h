#ifndef _INCL_RP_SEM
#define _INCL_RP_SEM

#include "stdlib.h"
#include <math.h>
#include "grid.h"
#include <string.h>

typedef struct rpSem{
    grid* g;
    double maxDepth;
    int nClasses;
    double* bgDist;
    long int* sz;
    int useProj;
} rpSem;

rpSem* rpSem_init(grid* g, double maxDepth, int nClasses, double* bgDist, int useProj);

void rpSem_forward(rpSem* rp, double *predsGeom, double* gradPredsGeom, double* predsSem, double* gradPredsSem, double* E_psi, double *origins, double*directions, double* depths, int* classIds, int bs, int nrays);

#endif
