#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef __pycbf_cpu_pyusel__
#define __pycbf_cpu_pyusel__

typedef float pycbf_datatype;

struct IntrpData1D_Fixed {
    int N;          // length of data in y
    pycbf_datatype xstart;   // starting point of xarray
    pycbf_datatype dx;       // sampling period in x
    pycbf_datatype * y;      // pointer to input data
    pycbf_datatype * y2;     // pointer to second derivative vector
    pycbf_datatype fill;     // fill value for out of bounds inteprolation requests
};

typedef struct IntrpData1D_Fixed IntrpData1D_Fixed;

extern pycbf_datatype * cubic1D_fixed(pycbf_datatype * xin, int Nin, IntrpData1D_Fixed * knots);
extern IntrpData1D_Fixed * tie_knots1D_fixed(pycbf_datatype * y, int N, pycbf_datatype dx, pycbf_datatype xstart, pycbf_datatype fill, int ycopy);
extern void free_IntrpData1D_Fixed(IntrpData1D_Fixed * knots);

// function declarations
extern void rxengine(int N, pycbf_datatype c, pycbf_datatype * ref, pycbf_datatype * points, pycbf_datatype *tau);
extern void pwtxengine(int N, pycbf_datatype c, pycbf_datatype tref, pycbf_datatype *ref, pycbf_datatype *norm, pycbf_datatype *points, pycbf_datatype *tau);
extern void genmask3D(int N, pycbf_datatype fmaj, int dynmaj, pycbf_datatype fmin, int dynmin, pycbf_datatype * n, pycbf_datatype *focus, pycbf_datatype *ref, pycbf_datatype *points, int *mask);
extern void calcindices(int Ntau, int Ntrace, pycbf_datatype tstart, pycbf_datatype fs, pycbf_datatype * tau, int *mask, int * tind);
extern void selectdata(int Ntind, int *tind, pycbf_datatype *data, pycbf_datatype *dataout);
extern void copysubvec(int Norig, int Nsub, int index, pycbf_datatype *orig, pycbf_datatype *sub);
extern void sumvecs(int N, pycbf_datatype *vec1, pycbf_datatype *vec2, pycbf_datatype v0, pycbf_datatype *summed);
extern void printifa(int i, pycbf_datatype f, pycbf_datatype * a, int na);
extern void fillarr(int N, pycbf_datatype *vec, pycbf_datatype fillval);

extern void beamform_cubic(
    pycbf_datatype t0,
    pycbf_datatype dt,
    int nt,
    pycbf_datatype * sig,
    int nout,
    pycbf_datatype thresh,
    pycbf_datatype * tautx,
    pycbf_datatype * apodtx,
    pycbf_datatype * taurx,
    pycbf_datatype * apodrx,
    pycbf_datatype * out
);

extern void beamform_nearest(
    pycbf_datatype t0,
    pycbf_datatype dt,
    int nt,
    pycbf_datatype * sig,
    int nout,
    pycbf_datatype thresh,
    pycbf_datatype * tautx,
    pycbf_datatype * apodtx,
    pycbf_datatype * taurx,
    pycbf_datatype * apodrx,
    pycbf_datatype * out,
    int usf
);

#endif