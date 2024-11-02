#include "pycbf.h"

void beamform(
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
) 
{
    // make the interpolator for the input signal
    IntrpData1D_Fixed * knots = tie_knots1D_fixed(sig, nt, dt, t0, 0.0f, 1);

    // sum the delay tabs
    float * tau = (float *) malloc(sizeof(float) * nout);
    sumvecs(nout, tautx, taurx, 0.0f, tau);

    // delay data to output points
    float * delayed = cubic1D_fixed(tau, nout, knots);

    // free datastructures when they are done
    free(tau);
    free_IntrpData1D_Fixed(knots);

    // apply apodization and sum with current output vector
    for(int i=0; i<nout; ++i) 
    {
        out[i] += delayed[i] * apodtx[i] * apodrx[i];
    }
}