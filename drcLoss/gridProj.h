#ifndef _INCL_GRID_PROJ
#define _INCL_GRID_PROJ

#include "grid.h"
typedef grid gridProj;

long int* gridProj_size(gridProj *g);

long int gridProj_nCells(gridProj *g);

int gridProj_cellIndex(gridProj *g, double *point, long int *inds);

int gridProj_initRayPoint(gridProj *g, double *origin, double *direction, double *initPoint, long int *initInds);

int gridProj_nextRayPoint(gridProj *g, double *point, double *direction, double *nextPoint, long int* nextInd);

int gridProj_traceRay(gridProj *g, double *origin, double *direction, long int *indSeq, double *depthSeq);
#endif
