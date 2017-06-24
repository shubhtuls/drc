#include <assert.h>
#include "rayPotGeom.h"
#include "gridReg.h"
#include "gridProj.h"
#include "rpUtils.h"

#include <stdio.h>
rpGeom* rpGeom_init(grid* g, double maxDepth, int useProj){
    rpGeom* rp;
    if((rp = malloc(sizeof *rp)) != NULL){
        rp->g = g;
        rp->maxDepth = maxDepth;
        rp->sz = useProj ? gridProj_size(g) : gridReg_size(g);
        rp->useProj = useProj;
    }
    return rp;
}

void rpGeom_forward(rpGeom* rp, double *predsGeom, double* gradPredsGeom, double* E_psi, double *origins, double*directions, double* depths, int bs, int nrays, int useMaskOnly){
    int Nd = rp->g->nDim;
    int maxHits = 0;
    for(int d = 0;d<Nd;d++)maxHits+=rp->sz[d];
    double* rayDepths = malloc(sizeof(double)*maxHits);
    long* raySubs = malloc(sizeof(long)*Nd*maxHits);

    long int* x_r = malloc(sizeof(long int)*maxHits);
    double* psi_r = malloc(sizeof(double)*(maxHits+1));
    int N_r, k_rev;
    double d_r, p_zr_i, grad_k_rev, prob_k_rev;
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
            d_r = depths[b*nrays + r];
            //printf("d_N_r = %f\n",rayDepths[N_r-1]);
            for(int ix=0;ix<N_r;ix++){
                 x_r[ix] = sub2ind(raySubs + Nd*ix, rp->sz, Nd);
                 prod_x[ix] = predsGeom[b*nCells + x_r[ix]] * ((ix > 0) ? prod_x[ix-1] : 1);
                 if(useMaskOnly) psi_r[ix] = d_r;
                 else psi_r[ix] = f_psi_obj(rayDepths[ix], d_r);
                 //printf("ix=%d, xr_ix = %ld, prod=%f, psi = %f\n",ix,x_r[ix],prod_x[ix],psi_r[ix]);
                 //printf("raySubs=%ld, %ld, %ld\n",raySubs[Nd*ix],raySubs[Nd*ix+1], raySubs[Nd*ix+2]);
            }
            if(useMaskOnly) psi_r[N_r] = 1-d_r;
            else psi_r[N_r] = f_psi_obj(rp->maxDepth, d_r);
            E_psi[b*nrays + r] = psi_r[0];

            for(int ix=0;ix<N_r;ix++){
                 E_psi[b*nrays + r] += (psi_r[ix+1] - psi_r[ix])*prod_x[ix];
                 p_zr_i = ((ix > 0) ? prod_x[ix-1] : 1) - prod_x[ix];

                 k_rev = N_r - ix - 1;
                 sumReverse_x[k_rev] = (psi_r[k_rev+1]-psi_r[k_rev])*prod_x[k_rev] + ((k_rev < N_r-1) ? sumReverse_x[k_rev+1] : 0);
                 prob_k_rev = predsGeom[b*nCells + x_r[k_rev]];
                 grad_k_rev = (prob_k_rev > 0) ? sumReverse_x[k_rev]/prob_k_rev : 0 ;
                 if(grad_k_rev == grad_k_rev) gradPredsGeom[b*nCells + x_r[k_rev]] += grad_k_rev;
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

void rpGeom_weighted_forward(rpGeom* rp, double *predsGeom, double* gradPredsGeom, double* E_psi, double *origins, double*directions, double* depths, double* weights, int bs, int nrays, int useMaskOnly){
    int Nd = rp->g->nDim;
    int maxHits = 0;
    for(int d = 0;d<Nd;d++)maxHits+=rp->sz[d];
    double* rayDepths = malloc(sizeof(double)*maxHits);
    long* raySubs = malloc(sizeof(long)*Nd*maxHits);

    long int* x_r = malloc(sizeof(long int)*maxHits);
    double* psi_r = malloc(sizeof(double)*(maxHits+1));
    int N_r, k_rev;
    double d_r, p_zr_i, grad_k_rev, prob_k_rev, wt_r;
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
            d_r = depths[b*nrays + r];
            wt_r = weights[b*nrays + r];
            //printf("d_N_r = %f\n",rayDepths[N_r-1]);
            for(int ix=0;ix<N_r;ix++){
                 x_r[ix] = sub2ind(raySubs + Nd*ix, rp->sz, Nd);
                 prod_x[ix] = predsGeom[b*nCells + x_r[ix]] * ((ix > 0) ? prod_x[ix-1] : 1);
                 if(useMaskOnly) psi_r[ix] = d_r;
                 else psi_r[ix] = f_psi_obj(rayDepths[ix], d_r);
                 psi_r[ix] *= wt_r;
                 //printf("ix=%d, xr_ix = %ld, prod=%f, psi = %f\n",ix,x_r[ix],prod_x[ix],psi_r[ix]);
                 //printf("raySubs=%ld, %ld, %ld\n",raySubs[Nd*ix],raySubs[Nd*ix+1], raySubs[Nd*ix+2]);
            }
            if(useMaskOnly) psi_r[N_r] = 1-d_r;
            else psi_r[N_r] = f_psi_obj(rp->maxDepth, d_r);
            psi_r[N_r] *= wt_r;
            E_psi[b*nrays + r] = psi_r[0];

            for(int ix=0;ix<N_r;ix++){
                 E_psi[b*nrays + r] += (psi_r[ix+1] - psi_r[ix])*prod_x[ix];
                 p_zr_i = ((ix > 0) ? prod_x[ix-1] : 1) - prod_x[ix];

                 k_rev = N_r - ix - 1;
                 sumReverse_x[k_rev] = (psi_r[k_rev+1]-psi_r[k_rev])*prod_x[k_rev] + ((k_rev < N_r-1) ? sumReverse_x[k_rev+1] : 0);
                 prob_k_rev = predsGeom[b*nCells + x_r[k_rev]];
                 grad_k_rev = (prob_k_rev > 0) ? sumReverse_x[k_rev]/prob_k_rev : 0 ;
                 if(grad_k_rev == grad_k_rev) gradPredsGeom[b*nCells + x_r[k_rev]] += grad_k_rev;
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

