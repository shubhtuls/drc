#ifndef _INCL_GRID_REG
#define _INCL_GRID_REG
#include "grid.h"
typedef grid gridReg;

long int* gridReg_size(gridReg *g);

long int gridReg_nCells(gridReg *g);

int gridReg_cellIndex(gridReg *g, double *point, long int *inds);

int gridReg_initRayPoint(gridReg *g, double *origin, double *direction, double *initPoint, long int *initInds);

int gridReg_nextRayPoint(gridReg *g, double *point, double *direction, double *nextPoint, long int* nextInd);

int gridReg_traceRay(gridReg *g, double *origin, double *direction, long int *indSeq, double *depthSeq);
#endif
