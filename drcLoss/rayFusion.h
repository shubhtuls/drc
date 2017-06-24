#ifndef _INCL_RP_FUSE
#define _INCL_RP_FUSE

#include "stdlib.h"
#include <math.h>
#include "grid.h"
#include <string.h>

typedef struct rpFuse{
    grid* g;
    long int* sz;
    int useProj;
} rpFuse;

rpFuse* rpFuse_init(grid* g, int useProj);

void rpFuse_forward(rpFuse* rp, double* emptyCount, double* occCount, double *origins, double*directions, double* depths, int bs, int nrays, int useMaskOnly);

#endif
