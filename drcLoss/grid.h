#ifndef _INCL_GRID
#define _INCL_GRID

static const double eps = 1e-4;
static const double inf = 1e10;

void addMult(double *p1, double *p2, double c, int Nd);

double distFunc(double *p1, double *p2, int Nd);

void copyArrInt(long int *to, long int *from, int Nd);

void copyArr(double *to, double *from, int Nd);

typedef struct grid{
    int nDim;
    double* minBounds;
    double* maxBounds;
    double* focal;
} grid;

grid* grid_init(int nDim, double* minBounds, double* maxBounds, double* focal);
#endif
