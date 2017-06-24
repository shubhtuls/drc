#include <assert.h>
#include "rayFusion.h"
#include "gridReg.h"
#include "gridProj.h"
#include "rpUtils.h"

#include <stdio.h>
rpFuse* rpFuse_init(grid* g, int useProj){
    rpFuse* rp;
    if((rp = malloc(sizeof *rp)) != NULL){
        rp->g = g;
        rp->sz = useProj ? gridProj_size(g) : gridReg_size(g);
        rp->useProj = useProj;
    }
    return rp;
}

void rpFuse_forward(rpFuse* rp, double *emptyCount, double* occCount, double *origins, double*directions, double* depths, int bs, int nrays, int useMaskOnly){
    int Nd = rp->g->nDim;
    int maxHits = 0;
    for(int d = 0;d<Nd;d++)maxHits+=rp->sz[d];
    double* rayDepths = malloc(sizeof(double)*maxHits);
    long* raySubs = malloc(sizeof(long)*Nd*maxHits);
    int N_r;
    long int* x_r = malloc(sizeof(long int)*maxHits);
    double d_r;
    long nCells = rp->useProj ? gridProj_nCells(rp->g) : gridReg_nCells(rp->g);

    int N_r_tot = 0;
    for(int b=0; b < bs; b++){
        for(int r=0; r < nrays; r++){
            //if(r<10) printf("b,r = (%d, %d\n)", b, r);
            long int startOffset = b*nrays*Nd + r*Nd;
            if(rp->useProj) N_r = gridProj_traceRay(rp->g, origins + startOffset, directions + startOffset, raySubs, rayDepths);
            else N_r = gridReg_traceRay(rp->g, origins + startOffset, directions + startOffset, raySubs, rayDepths);
            //if(r<10) printf("N_r = %d, origin = (%f,%f,%f), direction = (%f,%f,%f) \n",N_r,origins[startOffset+0],origins[startOffset+1],origins[startOffset+2],directions[startOffset+0],directions[startOffset+1],directions[startOffset+2]);
            assert(N_r <= maxHits);
            N_r_tot += N_r;
            if(N_r == 0) continue;
            //continue;
            d_r = depths[b*nrays + r];
            //if(r<10) printf("d_N_r = %f, d_r = %f \n",rayDepths[N_r-1],d_r);
            for(int ix=0;ix<N_r;ix++){
                x_r[ix] = sub2ind(raySubs + Nd*ix, rp->sz, Nd);
                // Free voxels
                if ((useMaskOnly && d_r==1) || (!useMaskOnly && (ix < (N_r-1)) && rayDepths[ix+1] < d_r)){
                    emptyCount[b*nCells + x_r[ix]] += 1;
                }
                // Occupied Voxel
                if(!useMaskOnly && (ix < (N_r-1)) && rayDepths[ix] < d_r && rayDepths[ix+1] >= d_r){
                    occCount[b*nCells + x_r[ix]] += 1;
                }
            }
        }
    }

    //printf("N_r_tot = %d\n",N_r_tot);
    free(rayDepths);
    free(raySubs);
    free(x_r);
    free(rp->sz);
    return;

}
