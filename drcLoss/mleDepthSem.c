#include <assert.h>
#include "mleDepthSem.h"
#include "gridReg.h"
#include "gridProj.h"
#include "rpUtils.h"

#include <stdio.h>
mleDepthSem* mleDepthSem_init(grid* g, double maxDepth, int useProj){
    mleDepthSem* rp;
    if((rp = malloc(sizeof *rp)) != NULL){
        rp->g = g;
        rp->maxDepth = maxDepth;
        rp->sz = useProj ? gridProj_size(g) : gridReg_size(g);
        rp->useProj = useProj;
    }
    return rp;
}


void mleDepthSem_forward(mleDepthSem* rp, double *predsGeom, int* predsSem, double *origins, double*directions, double* inferredDepth, int* inferredClass, int bs, int nrays){
    int Nd = rp->g->nDim;
    int maxHits = 0;
    for(int d = 0;d<Nd;d++)maxHits+=rp->sz[d];
    double* rayDepths = malloc(sizeof(double)*maxHits);
    long* raySubs = malloc(sizeof(long)*Nd*maxHits);

    long int* x_r = malloc(sizeof(long int)*maxHits);
    double* psi_r = malloc(sizeof(double)*(maxHits+1));
    int N_r, c_r, k_rev, maxInd;
    double d_r, p_zr_i, grad_k_rev, prob_k_rev, maxProb, probHit;
    long nCells = rp->useProj ? gridProj_nCells(rp->g) : gridReg_nCells(rp->g);

    double* prod_x = malloc(sizeof(double)*maxHits);
    double* sumReverse_x = malloc(sizeof(double)*maxHits);
    int N_r_tot = 0;

    for(int b=0; b < bs; b++){
        for(int r=0; r < nrays; r++){
            //printf("b,r = (%d, %d)", b, r);
            long int startOffset = b*nrays*Nd + r*Nd;
            if(rp->useProj) N_r = gridProj_traceRay(rp->g, origins + startOffset, directions + startOffset, raySubs, rayDepths);
            else N_r = gridReg_traceRay(rp->g, origins + startOffset, directions + startOffset, raySubs, rayDepths);
            //printf("N_r = %d, origin = (%f,%f,%f), direction = (%f,%f,%f) \n",N_r,origins[startOffset+0],origins[startOffset+1],origins[startOffset+2],directions[startOffset+0],directions[startOffset+1],directions[startOffset+2]);
            assert(N_r <= maxHits);
            N_r_tot += N_r;
            if(N_r == 0) continue;
            //continue;
            maxProb = 0; maxInd = 0;
            for(int ix=0;ix<N_r;ix++){
                 x_r[ix] = sub2ind(raySubs + Nd*ix, rp->sz, Nd);
                 prod_x[ix] = predsGeom[b*nCells + x_r[ix]] * ((ix > 0) ? prod_x[ix-1] : 1);
                 probHit = ((ix > 0) ? prod_x[ix-1] : 1) - prod_x[ix];
                 if(probHit > maxProb){maxProb = probHit; maxInd = ix;}
                 //printf("ix=%d, xr_ix = %ld, prod=%f, psi = %f\n",ix,x_r[ix],prod_x[ix],psi_r[ix]);
                 //printf("raySubs=%ld, %ld, %ld\n",raySubs[Nd*ix],raySubs[Nd*ix+1], raySubs[Nd*ix+2]);
            }
            if(maxProb > prod_x[N_r-1]){
                 inferredDepth[b*nrays + r] = rayDepths[maxInd];
                 inferredClass[b*nrays + r] = predsSem[b*nCells + x_r[maxInd]];
            }
            else{
                 inferredDepth[b*nrays + r] = rp->maxDepth;
                 inferredClass[b*nrays + r] = -1;
            }
        }
    }

    //printf("N_r_tot = %d\n",N_r_tot);
    free(rayDepths);
    free(raySubs);
    free(x_r);
    free(psi_r);
    free(prod_x);
    free(sumReverse_x);
    free(rp->sz);
    return;

}

