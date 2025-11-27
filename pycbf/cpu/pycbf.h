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