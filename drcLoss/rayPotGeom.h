#ifndef _INCL_RP_GEOM
#define _INCL_RP_GEOM

#include "stdlib.h"
#include <math.h>
#include "grid.h"
#include <string.h>

typedef struct rpGeom{
    grid* g;
    double maxDepth;
    long int* sz;
    int useProj;
} rpGeom;

rpGeom* rpGeom_init(grid* g, double maxDepth, int useProj);

void rpGeom_forward(rpGeom* rp, double *predsGeom, double* gradPredsGeom, double* E_psi, double *origins, double*directions, double* depths, int bs, int nrays, int useMaskOnly);
void rpGeom_forward_weighted(rpGeom* rp, double *predsGeom, double* gradPredsGeom, double* E_psi, double *origins, double*directions, double* depths, double* weights, int bs, int nrays, int useMaskOnly);

#endif
