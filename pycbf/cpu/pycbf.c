#include "pycbf.h"

/**
 * fillarr
 * fill a given vector with a given filvalue
 */
void fillarr(int N, pycbf_datatype *vec, pycbf_datatype fillval) {
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
void rxengine(int N, pycbf_datatype c, pycbf_datatype * ref, pycbf_datatype * points, pycbf_datatype *tau) {
    // define the output array of tau
    pycbf_datatype xdiff;
    pycbf_datatype ydiff;
    pycbf_datatype zdiff;

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
void pwtxengine(int N, pycbf_datatype c, pycbf_datatype tref, pycbf_datatype *ref, pycbf_datatype *norm, pycbf_datatype *points, pycbf_datatype *tau) {
    // iterate through each point
    pycbf_datatype xdiff;
    pycbf_datatype ydiff;
    pycbf_datatype zdiff;

    for(int i = 0; i < N; ++i) {
        xdiff = norm[0] * (points[3*i+0] - ref[0]);
        ydiff = norm[1] * (points[3*i+1] - ref[1]);
        zdiff = norm[2] * (points[3*i+2] - ref[2]);

        tau[i] += (xdiff + ydiff + zdiff)/c + tref;
    }
}

void genmask3D(int N, pycbf_datatype fmaj, int dynmaj, pycbf_datatype fmin, int dynmin, pycbf_datatype * n, pycbf_datatype *focus, pycbf_datatype *ref, pycbf_datatype *points, int *mask) {
    pycbf_datatype nmaj[3] = {n[0], n[1], 0.0f};
    pycbf_datatype nmin[3] = {n[1], -n[0], 0.0f};
    pycbf_datatype rmaj;
    pycbf_datatype rmin;
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
void sumvecs(int N, pycbf_datatype *vec1, pycbf_datatype *vec2, pycbf_datatype v0, pycbf_datatype *summed) {
    for (int i = 0; i < N; ++i) {
        summed[i] = vec1[i] + vec2[i] + v0;
    }
}

/**
 * calcindices
 * returns an Ntau length array of integer indices with values ranging from [-1, Ntrace)
 * A value of -1 indicates an array index out of bounds or a masked out value
 */
void calcindices(int Ntau, int Ntrace, pycbf_datatype tstart, pycbf_datatype fs, pycbf_datatype * tau, int *mask, int * tind) {
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
void selectdata(int Ntind, int *tind, pycbf_datatype *data, pycbf_datatype *dataout) {
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
void copysubvec(int Norig, int Nsub, int index, pycbf_datatype *orig, pycbf_datatype *sub) {
    for (int i = 0; (i < Nsub) && (Nsub*index+i < Norig); ++i) {
        sub[i] = orig[Nsub*index+i];
    }
}

/**
 * 
 */
void printifa(int i, pycbf_datatype f, pycbf_datatype * a, int na) {
    printf("%p \n", a);
    printf("%d, %0.03f, [", i, f);
    for (int icount = 0; icount < na; ++icount) {
        printf("%f, ", a[icount]);
    }
    printf("\b\b]\n");
}

/**
 * cubic1D_fixed: 1D cubic interpolator for a fixed sampling scheme
 * Based on the cubic interpolation algorithm from Numerical Recipes
 * Comments are to the writer's knowledge
 * Wren Wightman, wewightman@github.com, wew@gitlab.oit.duke.edu/, 2023
 * 
 * Parameters:
 *  xin:    pointer to vector of intperpolation coordinates
 *  Nin:    number of input coordinates
 *  knots:  knot structure with interpolation parameters
*/
pycbf_datatype * cubic1D_fixed(pycbf_datatype * xin, int Nin, IntrpData1D_Fixed * knots) 
{
    // loop variables and return buffer
    pycbf_datatype x, a, b, c, d, xlead, xlag;
    int j;

    pycbf_datatype * y = (pycbf_datatype *) malloc(sizeof(pycbf_datatype) * Nin);

    // procedurally define bounds
    pycbf_datatype dx = knots->dx;
    pycbf_datatype xmin = knots->xstart;
    pycbf_datatype xmax = xmin + dx * (pycbf_datatype)((knots->N) - 1);

    for(int i=0; i<Nin; ++i) 
    {
        x = xin[i];                 // extract interpolation point
        j = (int) ((x-xmin)/dx);    // calculate knot index

        // if out of bounds, return fill vlaue
        if ((x < xmin) || (x > xmax))
        {
            y[i] = knots->fill;
        }

        // interpolate in bounds
        else
        {
            // extract x value
            xlead = xmin + dx * (int)(j+1); // calculate high x
            xlag = xmin + dx * (int)(j);    // calculate low x

            // generate interpolation terms
            a = (xlead - x) / dx;
            b = (x - xlag) / dx;
            c = (1.0f/6.0f) * (powf(a, 3.0f) - a) * powf(dx, 2.0f);
            d = (1.0f/6.0f) * (powf(b, 3.0f) - b) * powf(dx, 2.0f);

            // calculate y value
            y[i] = a*(knots->y)[j] + b*knots->y[j+1] + c*knots->y2[j] + d*knots->y2[j+1];
        }
    }
    return y;
}

/**
 * cubic1D_fixed: 1D cubic interpolator for a fixed sampling scheme
 * Based on the cubic interpolation algorithm from Numerical Recipes
 * Comments are to the writer's knowledge
 * Wren Wightman, wewightman@github.com, wew@gitlab.oit.duke.edu/, 2023
 * 
 * Parameters:
 *  xin:    pointer to vector of intperpolation coordinates
 *  knots:  knot structure with interpolation parameters
*/
pycbf_datatype cubic1D_fixed_point(pycbf_datatype xin, IntrpData1D_Fixed * knots) 
{
    // loop variables and return buffer
    pycbf_datatype x, a, b, c, d, xlead, xlag;
    int j;

    // procedurally define bounds
    pycbf_datatype dx = knots->dx;
    pycbf_datatype xmin = knots->xstart;
    pycbf_datatype xmax = xmin + dx * (pycbf_datatype)((knots->N) - 1);


    x = xin;                    // extract interpolation point
    j = (int) ((x-xmin)/dx);    // calculate knot index

    // if out of bounds, return fill vlaue
    if ((x < xmin) || (x > xmax))
    {
        return knots->fill;
    }

    // interpolate in bounds
    else
    {
        // extract x value
        xlead = xmin + dx * (int)(j+1); // calculate high x
        xlag = xmin + dx * (int)(j);    // calculate low x

        // generate interpolation terms
        a = (xlead - x) / dx;
        b = (x - xlag) / dx;
        c = (1.0f/6.0f) * (powf(a, 3.0f) - a) * powf(dx, 2.0f);
        d = (1.0f/6.0f) * (powf(b, 3.0f) - b) * powf(dx, 2.0f);

        // calculate y value
        return a*(knots->y)[j] + b*knots->y[j+1] + c*knots->y2[j] + d*knots->y2[j+1];
    }
}


/**
 * tie_knots1D_fixed: Generate second-derivative knots for a fixed sampling scheme
 * Based on the cubic interpolation algorithm from Numerical Recipes
 * Comments are to the writer's knowledge
 * Wren Wightman, github:@wewightman, 2023
*/
IntrpData1D_Fixed * tie_knots1D_fixed(pycbf_datatype * y, int N, pycbf_datatype dx, pycbf_datatype xstart, pycbf_datatype fill, int ycopy) {
    // generate loop variables and buffers
    pycbf_datatype * y2 = (pycbf_datatype *) malloc(sizeof(pycbf_datatype) * N);
    pycbf_datatype * u  = (pycbf_datatype *) malloc(sizeof(pycbf_datatype) * N);

    pycbf_datatype sig, p;

    // set y'' = 0 boundary condition
    y2[0] = 0.0f;
    u[0] = 0.0f;

    // forward decomposition loop of tri-diagonal inversion algorithm
    for (int i=1; i < N-1; ++i)
    {
        sig = 0.5f;
        p = sig*y2[i-1]+2.0f;
        y2[i] = (sig-1.0f)/p;
        u[i] = (y[i+1] - y[i])/dx - (y[i] - y[i-1])/dx;
        u[i] = (6.0f*u[i] / (2.0f*dx) - sig * u[i-1]) / p;
    }

    // backsubstitution loop of tridiagonal algorithm
    y2[N-1] = 0.0f;

    for (int k = N-2; k >= 0; k--) 
    {
        y2[k] = y2[k] * y2[k+1] + u[k];
    }

    free(u); // free loop buffer

    // return a filled knots structure
    IntrpData1D_Fixed * knots = (IntrpData1D_Fixed *) malloc(sizeof(IntrpData1D_Fixed));
    knots->dx = dx;
    knots->xstart = xstart;
    knots->N = N;
    knots->fill = fill;
    knots->y2 = y2;

    // make a deep copy of y if requested, otherwise, save reference to input y
    if (ycopy)
    {
        pycbf_datatype * ytilde = (pycbf_datatype *) malloc(sizeof(pycbf_datatype) * N);
        for(int i = 0; i < N; ++i) ytilde[i] = y[i];
        knots->y = ytilde;
    }
    else 
    {
        knots->y = y;
    }

    return knots;
}

/**
 * Free an instance of an IntrpData1D_Fixed structure
*/
void free_IntrpData1D_Fixed(IntrpData1D_Fixed * knots)
{
    // free all generated vectors and free the space allocated for the structure
    free(knots->y);
    free(knots->y2);
    free(knots);
}

void beamform_cubic(
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
) 
{
    // make the interpolator for the input signal
    IntrpData1D_Fixed * knots = tie_knots1D_fixed(sig, nt, dt, t0, 0.0f, 1);

    // apply apodization and sum with current output vector
    pycbf_datatype apod;
    for(int i=0; i<nout; ++i) 
    {
        apod = apodtx[i] * apodrx[i];
        if (fabs(apod) >= thresh) 
        {
            out[i] += apod * cubic1D_fixed_point(tautx[i] + taurx[i], knots);
        }
    }
    free_IntrpData1D_Fixed(knots);
}

void beamform_nearest(
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
) 
{
    // upsample the signal if usf is larger than 1
    pycbf_datatype * sig_usf;
    int nt_usf;
    pycbf_datatype dt_usf;
    if (usf == 1)
    {
        sig_usf = sig;
        nt_usf  = nt;
        dt_usf  = dt;
    }
    else
    {
        // figure out upsampled dimensions and spacing
        nt_usf = usf * (nt-1) + 1;
        dt_usf = dt / ((pycbf_datatype) usf);

        // make the time vector to interpolate along
        pycbf_datatype * tin_usf = (pycbf_datatype *) malloc(nt_usf * sizeof(pycbf_datatype));
        for (int it=0; it<nt_usf; ++it) tin_usf[it] = t0 + it*dt_usf;

        // make the interpolator for the input signal
        IntrpData1D_Fixed * knots = tie_knots1D_fixed(sig, nt, dt, t0, 0.0f, 1);

        // upsample the signal
        sig_usf = cubic1D_fixed(tin_usf, nt_usf, knots);

        // free the vector
        free_IntrpData1D_Fixed(knots);
        free(tin_usf);
    }

    // apply apodization and sum with current output vector
    pycbf_datatype apod;
    int it;
    for(int i=0; i<nout; ++i) 
    {
        // calculate apodization
        apod = apodtx[i] * apodrx[i];

        // calculate time index 
        it = (int) ((tautx[i] + taurx[i] - t0)/dt_usf + 0.5);

        // run nearest neighbor interpolation
        if ((fabs(apod) >= thresh) && ((it >= 0) && (it < nt_usf))) out[i] += apod * sig_usf[it];
    }

    if (usf != 1) free(sig_usf);
}