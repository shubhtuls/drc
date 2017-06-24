#include "rpUtils.h"
#include "math.h"
double f_psi_scene(double d, double d_r){
    double val = fabs(1.0/d - 1.0/d_r);
    //printf("val = %f, d = %f, d_r = %f\n",val,1/d,1/d_r);
    return (val < 2) ? val : 2;
}

double f_psi_obj(double d, double d_r){
    double val = fabs(d - d_r);
    //printf("val = %f, d = %f, d_r = %f\n",val,1/d,1/d_r);
    return val;
}

double psi_func_sem(double d, double d_r, double* prob, int c){
    return (f_psi_scene(d, d_r) - 0.1*log(prob[c]));
}

long sub2ind(long int* sub, long int* sz, int Nd){
    long ind = 0;
    for(int d=0; d< Nd; d++) ind = ind*sz[d] + sub[d];
    return ind;
}
