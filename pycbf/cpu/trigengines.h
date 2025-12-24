// function declarations
extern void rxengine(int N, float c, float * ref, float * points, float *tau);
extern void pwtxengine(int N, float c, float tref, float *ref, float *norm, float *points, float *tau);
extern void genmask3D(int N, float fmaj, int dynmaj, float fmin, int dynmin, float * n, float *focus, float *ref, float *points, int *mask);
extern void calcindices(int Ntau, int Ntrace, float tstart, float fs, float * tau, int *mask, int * tind);
extern void selectdata(int Ntind, int *tind, float *data, float *dataout);
extern void copysubvec(int Norig, int Nsub, int index, float *orig, float *sub);
extern void sumvecs(int N, float *vec1, float *vec2, float v0, float *summed);
extern void printifa(int i, float f, float * a, int na);
extern void fillarr(int N, float *vec, float fillval);