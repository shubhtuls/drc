#ifndef _INCL_RP_COLOR
#define _INCL_RP_COLOR

#include "stdlib.h"
#include <math.h>
#include "grid.h"
#include <string.h>

typedef struct rpColor{
    grid* g;
    double* bgColor;
    long int* sz;
    int useProj;
} rpColor;

rpColor* rpColor_init(grid* g, double* bgColor, int useProj);

void rpColor_forward(rpColor* rp, double *predsGeom, double* gradPredsGeom, double* predsColor, double* gradPredsColor, double* E_psi, double *origins, double *directions, double* colors, int bs, int nrays);

#endif