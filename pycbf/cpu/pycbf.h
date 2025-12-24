#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef __cubic_interp_pyusel__
#define __cubic_interp_pyusel__

struct IntrpData1D_Fixed {
    int N;          // length of data in y
    float xstart;   // starting point of xarray
    float dx;       // sampling period in x
    float * y;      // pointer to input data
    float * y2;     // pointer to second derivative vector
    float fill;     // fill value for out of bounds inteprolation requests
};

typedef struct IntrpData1D_Fixed IntrpData1D_Fixed;

extern float * cubic1D_fixed(float * xin, int Nin, IntrpData1D_Fixed * knots);
extern IntrpData1D_Fixed * tie_knots1D_fixed(float * y, int N, float dx, float xstart, float fill, int ycopy);
extern void free_IntrpData1D_Fixed(IntrpData1D_Fixed * knots);

#endif

#ifndef __trigengines_pyusel__
#define __trigengines_pyusel__

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

#endif

#ifndef __pycbf_pyusel__
#define __pycbf_pyusel__

extern void beamform_cubic(
    float t0,
    float dt,
    int nt,
    float * sig,
    int nout,
    float thresh,
    float * tautx,
    float * apodtx,
    float * taurx,
    float * apodrx,
    float * out
);

extern void beamform_nearest(
    float t0,
    float dt,
    int nt,
    float * sig,
    int nout,
    float thresh,
    float * tautx,
    float * apodtx,
    float * taurx,
    float * apodrx,
    float * out,
    int usf
);

#endif