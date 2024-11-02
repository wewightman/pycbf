#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<malloc.h>
#include<math.h>
#include"trigengines.h"

/**
 * fillarr
 * fill a given vector with a given filvalue
 */
void fillarr(int N, float *vec, float fillval) {
    for (int i = 0; i < N; ++i) {
        vec[i] = fillval;
    }
}

/**
 * rxengine
 * Calculate the temporal distance from reference point to each point in the field
 * N: number of points in field
 * c: speed of sound [m/s]
 * xref, yref, zref: (x, y, z) coordinate of reference point [m]
 * xfield, yfield, zfield: pointers to length N arrays of (x, y, z) coordinates in field
 */
void rxengine(int N, float c, float * ref, float * points, float *tau) {
    // define the output array of tau
    float xdiff;
    float ydiff;
    float zdiff;

    // iterate through each point
    for(int i = 0; i < N; ++i) {
        xdiff = points[3*i+0] - ref[0];
        ydiff = points[3*i+1] - ref[1];
        zdiff = points[3*i+2] - ref[2];

        tau[i] += sqrtf(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff)/c;
    }
}

/**
 * pwtxengine
 * Calculate the temporal distance from reference point to each point in the field
 * N: number of points in field
 * c: speed of sound [m/s]
 * ref: (x, y, z) coordinate of reference point [m]
 * norm: (x, y, z) normal vector
 * xfield, nx, yfield, ny, zfield, nz: pointers to length N arrays of (x, y, z) coordinates in field
 */
void pwtxengine(int N, float c, float tref, float *ref, float *norm, float *points, float *tau) {
    // iterate through each point
    float xdiff;
    float ydiff;
    float zdiff;

    for(int i = 0; i < N; ++i) {
        xdiff = norm[0] * (points[3*i+0] - ref[0]);
        ydiff = norm[1] * (points[3*i+1] - ref[1]);
        zdiff = norm[2] * (points[3*i+2] - ref[2]);

        tau[i] += (xdiff + ydiff + zdiff)/c + tref;
    }
}

void genmask3D(int N, float fmaj, int dynmaj, float fmin, int dynmin, float * n, float *focus, float *ref, float *points, int *mask) {
    float nmaj[3] = {n[0], n[1], 0.0f};
    float nmin[3] = {n[1], -n[0], 0.0f};
    float rmaj;
    float rmin;
    int inmaj;
    int inmin;

    for(int i = 0; i < N; ++i) {
        // calculate radius from center line
        rmaj = nmaj[0] * (points[3*i+0] - ref[0]) + nmaj[1] * (points[3*i+1] - ref[1]);
        rmin = nmin[0] * (points[3*i+0] - ref[0]) + nmin[1] * (points[3*i+1] - ref[1]);
        if (rmaj < 0.0f) {rmaj = -rmaj;}
        if (rmin < 0.0f) {rmin = -rmin;}

        // determine if within major axis
        inmaj = 0;
        if(0 != dynmaj) {
            if (2.0f*rmaj <= (points[3*i+2] - ref[2])/fmaj) {inmaj=1;}
        } else {
            if (2.0f*rmaj <= (focus[2] - ref[2])/fmaj) {inmaj=-1;}
        }

        // calculate if within minor axis
        inmin = 0;
        if(0 != dynmin) {
            if (2*rmin <= (points[3*i+2] - ref[2])/fmin) {inmin=1;}
        } else {
            if (2*rmin <= (focus[2] - ref[2])/fmin) {inmin=1;}
        }

        mask[i] = inmaj && inmin;
    }
}

/**
 * sumvecs:
 * sum vec1, vec2, and a given constant value
 */
void sumvecs(int N, float *vec1, float *vec2, float v0, float *summed) {
    for (int i = 0; i < N; ++i) {
        summed[i] = vec1[i] + vec2[i] + v0;
    }
}

/**
 * calcindices
 * returns an Ntau length array of integer indices with values ranging from [-1, Ntrace)
 * A value of -1 indicates an array index out of bounds or a masked out value
 */
void calcindices(int Ntau, int Ntrace, float tstart, float fs, float * tau, int *mask, int * tind) {
    int index;

    for (int i = 0; i < Ntau; ++i) {
        index = (int) ((tau[i] - tstart)*fs);
        if (!mask[i]) {
            tind[i] = -1;
        } else if((index >= Ntrace) || (index < 0)) {
            tind[i] = -1;
            mask[i] = 0;
        } else {
            tind[i] = index;
        }
        
    }
}

/**
 * selectdata
 * reads rawdata from a given pointer for entires dataset
*/
void selectdata(int Ntind, int *tind, float *data, float *dataout) {
    int itind;
    for (int i = 0; i < Ntind; ++i) {
        itind = tind[i];
        if (itind < 0) {dataout[i] = 0;}
        else {dataout[i] = data[itind];}
    }
}

/**
 * copysubsvec
 * copies subsection of large vector to smaller vector
 * Norig - number of elements in the original vector
 * Nsub - number of elements in the sub vector
 * index - multiplicative offset of subvector
 * orig - pointer to large, original array
 * sub - pointer to buffer to store Nsmall elements
 */
void copysubvec(int Norig, int Nsub, int index, float *orig, float *sub) {
    for (int i = 0; (i < Nsub) && (Nsub*index+i < Norig); ++i) {
        sub[i] = orig[Nsub*index+i];
    }
}

/**
 * 
 */
void printifa(int i, float f, float * a, int na) {
    printf("%p \n", a);
    printf("%d, %0.03f, [", i, f);
    for (int icount = 0; icount < na; ++icount) {
        printf("%f, ", a[icount]);
    }
    printf("\b\b]\n");
}
