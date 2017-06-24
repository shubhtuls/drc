#ifndef _INCL_RP_UTIL
#define _INCL_RP_UTIL
double f_psi_scene(double d, double d_r);

double f_psi_obj(double d, double d_r);

double psi_func_sem(double d, double d_r, double* prob, int c);

long sub2ind(long int* sub, long int* sz, int Nd);
#endif
