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
