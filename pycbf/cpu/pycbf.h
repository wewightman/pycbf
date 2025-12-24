#include "cubic.h"
#include "trigengines.h"
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void beamform(
    float t0,
    float dt,
    int nt,
    float * sig,
    int nout,
    float * tautx,
    float * apodtx,
    float * taurx,
    float * apodrx,
    float * out
);