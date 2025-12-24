extern "C" {

    /**
     * nearest_interp: nearest neighbor interpolation
     */
    float nearest_interp(
        const float x0,     // starting position of the regularly spaced coordinate vector
        const float dx,     // spacing of the coordinate vector
        const int nx,       // number of points in the coordinate vector
        const float* y,     // values of of the function sampled on x
        float xout,         // coordintate to interpolate at
        float fill          // value to fill if out of bounds
    ) 
    {
        int ixo = (int) ((xout - x0)/dx + 0.5f);

        if ((ixo < 0) || (ixo >= nx)) return fill;

        return y[ixo];
    }

    /**
     * linear_interp: linear interpolation
     */
    float linear_interp(
        const float x0,     // starting position of the regularly spaced coordinate vector
        const float dx,     // spacing of the coordinate vector
        const int nx,       // number of points in the coordinate vector
        const float* y,     // values of of the function sampled on x
        float xout,         // coordintate to interpolate at
        float fill          // value to fill if out of bounds
    ) 
    {
        int ixo = (int) ((xout - x0)/dx);

        if (xout == x0 + dx * nx) return y[nx-1];

        if ((ixo < 0) || (ixo >= nx)) return fill;
        
        float frac = (xout - ixo * dx - x0)/dx;
        return y[ixo] * (1.0f - frac) + y[ixo] * frac;
    }

    /**
     * makima_interp: cubeic interpolation assuming regular spacing using the makima method
     */
    float makima_interp(
        const float x0,     // starting position of the regularly spaced coordinate vector
        const float dx,     // spacing of the coordinate vector
        const int nx,       // number of points in the coordinate vector
        const float* y,     // values of of the function sampled on x
        float xout,         // coordintate to interpolate at
        float fill          // value to fill if out of bounds
    ) 
    {
        float xn = x0 + dx * (nx-1);
        
        // boundary condition (bc) - exactly last sampled point
        if (xout == xn) return y[nx-1];

        // bc - out of bounds, use fill value
        else if ((xout < x0) || (xout > xn)) return fill;
                                
        int ixo = (int) ((xout - x0)/dx);
        float mm2, mm1, mp0, mp1, mp2, w0, w1, sp0, sp1, a, b, c, d, delta;

        // bc - first point
        if (ixo == 0) {
            mp0 = y[ixo+1] - y[ixo+0];
            mp1 = y[ixo+2] - y[ixo+1];

            sp0 = mp0;
            sp1 = (mp0 + mp1)/2;

        } 
        // bc - second point
        else if (ixo == 1) {
            mm1 = y[ixo+0] - y[ixo-1];
            mp0 = y[ixo+1] - y[ixo-0];
            mp1 = y[ixo+2] - y[ixo+1];
            mp2 = y[ixo+3] - y[ixo+2];

            sp0 = (mp0 + mm1)/2;

            if ((mm1 == mp0) && (mp0 == mp1) && (mp1 == mp2)) sp1 = 0;
            else {
                w0  = abs(mp2 - mp1) + abs(mp2 + mp1)/2;
                w1  = abs(mp0 - mm1) + abs(mp0 + mm1)/2;
                sp1 = (w0 * mp0 + w1 * mp1) / (w0 + w1);
            }
            

        } 
        // bc - third to last point
        else if (ixo == nx-3) {
            mm2 = y[ixo+-1] - y[ixo-2];
            mm1 = y[ixo+0] - y[ixo-1];
            mp0 = y[ixo+1] - y[ixo-0];
            mp1 = y[ixo+2] - y[ixo+1];

            if ((mm2 == mm1) && (mm1 == mp0) && (mp0 == mp1)) sp0 = 0;
            else {
                w0  = abs(mp1 - mp0) + abs(mp1 + mp0)/2;
                w1  = abs(mm1 - mm2) + abs(mm1 + mm2)/2;
                sp0 = (w0 * mm1 + w1*mp0) / (w0 + w1);
            }

            sp1 = (mp0 + mp1)/2;
        }
        // bc - second to last point
        else if (ixo == nx-2) {
            mm1 = y[ixo+0] - y[ixo-1];
            mp0 = y[ixo+1] - y[ixo+0];

            sp0 = (mm1+mp0)/2;

            sp1 = mp0;
        }
        // all other points
        else {
            mm2 = y[ixo-1] - y[ixo-2];
            mm1 = y[ixo+0] - y[ixo-1];
            mp0 = y[ixo+1] - y[ixo+0];
            mp1 = y[ixo+2] - y[ixo+1];
            mp2 = y[ixo+3] - y[ixo+2];

            if ((mm2 == mm1) && (mm1 == mp0) && (mp0 == mp1)) sp0 = 0;
            else {
                w0  = abs(mp1 - mp0) + abs(mp1 + mp0)/2;
                w1  = abs(mm1 - mm2) + abs(mm1 + mm2)/2;
                sp0 = (w0 * mm1 + w1*mp0) / (w0 + w1);
            }

            if ((mm1 == mp0) && (mp0 == mp1) && (mp1 == mp2)) sp1 = 0;
            else {
                w0  = abs(mp2 - mp1) + abs(mp2 + mp1)/2;
                w1  = abs(mp0 - mm1) + abs(mp0 + mm1)/2;
                sp1 = (w0 * mp0 + w1*mp1) / (w0 + w1);
            }
        }

        a = y[ixo];
        b = sp0;
        c = (3*mp0 - 2*sp0 - sp1)/dx;
        d = (sp0 + sp1 - 2*mp0)/(dx*dx);

        delta = xout - (x0 + dx * ixo);

        return a + b * delta + c * delta * delta + d * delta * delta * delta;
    }

    /**
     * akima_interp: cubic interpolation assuming regular spacing using the akima method
     */
    float akima_interp(
        const float x0,     // starting position of the regularly spaced coordinate vector
        const float dx,     // spacing of the coordinate vector
        const int nx,       // number of points in the coordinate vector
        const float* y,     // values of of the function sampled on x
        float xout,         // coordintate to interpolate at
        float fill          // value to fill if out of bounds
    ) 
    {
        float xn = x0 + dx * (nx-1);
        
        // boundary condition (bc) - exactly last sampled point
        if (xout == xn) return y[nx-1];

        // bc - out of bounds, use fill value
        else if ((xout < x0) || (xout > xn)) return fill;
                                
        int ixo = (int) ((xout- x0)/dx);
        float mm2, mm1, mp0, mp1, mp2, w0, w1, sp0, sp1, a, b, c, d, delta;

        // bc - first point
        if (ixo == 0) {
            mp0 = y[ixo+1] - y[ixo+0];
            mp1 = y[ixo+2] - y[ixo+1];

            sp0 = mp0;
            sp1 = (mp0 + mp1)/2;

        } 
        // bc - second point
        else if (ixo == 1) {
            mm1 = y[ixo+0] - y[ixo-1];
            mp0 = y[ixo+1] - y[ixo-0];
            mp1 = y[ixo+2] - y[ixo+1];
            mp2 = y[ixo+3] - y[ixo+2];

            sp0 = (mp0 + mm1)/2;

            if ((mm1 == mp0) && (mp0 == mp1) && (mp1 == mp2)) sp1 = 0;
            else {
                w0  = abs(mp2 - mp1);
                w1  = abs(mp0 - mm1);
                sp1 = (w0 * mp0 + w1 * mp1) / (w0 + w1);
            }
            

        } 
        // bc - third to last point
        else if (ixo == nx-3) {
            mm2 = y[ixo+-1] - y[ixo-2];
            mm1 = y[ixo+0] - y[ixo-1];
            mp0 = y[ixo+1] - y[ixo-0];
            mp1 = y[ixo+2] - y[ixo+1];

            if ((mm2 == mm1) && (mm1 == mp0) && (mp0 == mp1)) sp0 = 0;
            else {
                w0  = abs(mp1 - mp0);
                w1  = abs(mm1 - mm2);
                sp0 = (w0 * mm1 + w1*mp0) / (w0 + w1);
            }

            sp1 = (mp0 + mp1)/2;
        }
        // bc - second to last point
        else if (ixo == nx-2) {
            mm1 = y[ixo+0] - y[ixo-1];
            mp0 = y[ixo+1] - y[ixo+0];

            sp0 = (mm1+mp0)/2;

            sp1 = mp0;
        }
        // all other points
        else {
            mm2 = y[ixo-1] - y[ixo-2];
            mm1 = y[ixo+0] - y[ixo-1];
            mp0 = y[ixo+1] - y[ixo+0];
            mp1 = y[ixo+2] - y[ixo+1];
            mp2 = y[ixo+3] - y[ixo+2];

            if ((mm2 == mm1) && (mm1 == mp0) && (mp0 == mp1)) sp0 = 0;
            else {
                w0  = abs(mp1 - mp0);
                w1  = abs(mm1 - mm2);
                sp0 = (w0 * mm1 + w1*mp0) / (w0 + w1);
            }

            if ((mm1 == mp0) && (mp0 == mp1) && (mp1 == mp2)) sp1 = 0;
            else {
                w0  = abs(mp2 - mp1);
                w1  = abs(mp0 - mm1);
                sp1 = (w0 * mp0 + w1*mp1) / (w0 + w1);
            }
        }

        a = y[ixo];
        b = sp0;
        c = (3*mp0 - 2*sp0 - sp1)/dx;
        d = (sp0 + sp1 - 2*mp0)/(dx*dx);

        delta = xout - (x0 + dx * ixo);

        return a + b * delta + c * delta * delta + d * delta * delta * delta;
    }

    float (*jumpTable[])(
        const float x0,     // starting position of the regularly spaced coordinate vector
        const float dx,     // spacing of the coordinate vector
        const int nx,       // number of points in the coordinate vector
        const float* y,     // values of of the function sampled on x
        float xout,         // coordintate to interpolate at
        float fill          // value to fill if out of bounds
    ) = {
        nearest_interp,
        linear_interp,
        akima_interp,
        makima_interp
    };

    /**
     * korder_cubic_interp: cubic interpolation assuming regular spacing using korder fast cubic spline interpolation, described in this reference.
     * Requires a precomputed matrix S which calculates the derivatives of the 1D signal using a length k kernel.
     * k must be an even integer - should be checked before passing.
     * 
     * [1] S. K. Præsius and J. Arendt Jensen, “Fast Spline Interpolation using GPU Acceleration,” in 2024 IEEE Ultrasonics, Ferroelectrics, and Frequency Control Joint Symposium (UFFC-JS), Sep. 2024, pp. 1–5. doi: 10.1109/UFFC-JS60046.2024.10793976.
     */
    float korder_cubic_interp(
        const float x0,     // starting position of the regularly spaced coordinate vector
        const float dx,     // spacing of the coordinate vector
        const int nx,       // number of points in the coordinate vector
        const float* y,     // values of of the function sampled on x
        const float* S,     // a matrix containing values to calcualte the derivatives of y
        const int k,        // the length of the kernel and size of S is k by k
        float xout,         // coordintate to interpolate at
        float fill          // value to fill if out of bounds
    ) 
    {
        float y0, y1, yp0, yp1, xmax, xnorm, a0, a1, a2, a3;
        int ixin, ik0, iy0, i;

        ixin = (xout - x0)/dx;
        xmax = x0 + dx * (nx-1);

        // If out of bounds, use the fill value
        if ((xout < x0) || (ixin >= nx) || (xout > xmax)) return fill;

        // Calculate the starting index for the interpolation kernel
        if (ixin < k/2)
        {
            ik0 = ixin;
            iy0 = 0;
        }
        else if (ixin > nx - k/2 - 1)
        {
            ik0 = k - nx + ixin;
            iy0 = nx - k;
        }
        else
        {
            ik0 = (k-1)/2;
            iy0 = ixin - (k-1)/2;
        }

        // extract bounding y values
        y0 = y[ixin];
        y1 = y[ixin+1];

        // calculate the bounding derivatives
        yp0 = 0.0; 
        yp1 = 0.0;

        for (i = 0; i < k; ++i)
        {
            yp0 += S[ik0 * k     + i] * y[iy0 + i];
            yp1 += S[ik0 * k + k + i] * y[iy0 + i];
        }

        // calculate normalized x values
        xnorm = (xout - x0 - dx * ixin) / dx;

        // calculate the corefficients of the cubic polynomial
        a0 = y0;
        a1 = yp0;
        a2 = 3 * (y1 - y0) - 2 * yp0 - yp1;
        a3 = 2 * (y0 - y1) +     yp0 + yp1;

        // interpolate the point
        return a0 + a1 * xnorm + a2 * xnorm * xnorm + a3 * xnorm * xnorm * xnorm;
    }

    /**
     * xInfo: struct defining the bounds and spacing of a regularly spaced array
     */
    struct xInfo {
        float x0;   // the starting point of the vector
        float dx;   // the spacing between points
        int nx;     // the number of points in the vector
    };

    /**
     * RFInfo: struct defining the meadata of RF data
     */
    struct RFInfo {
        int ntx;            // the number of transmit events
        int nrx;            // the number of recieve events
        int ndim;           // the number of dimensions to beamform over
        struct xInfo tInfo; // the sampling information about the time vector
    };

    /**
     * calc_tautx_apodtx: calcualte the tx delay tabs and apodizations given transmit data structures
     * 
     * tau and apod are pointers to be filled with the correct values
     */
    void calc_tautx_apodtx(
        const int    ndim,  // 2 or 3 dimensions
        const float*  foc,  // focal spot
        const float* nvec,  // normal vector of wave propagation
        const float    c0,  // assumed speed of sound in media
        const float    t0,  // the time at which the wave reaches foc
        const float   ala,  // the acceptance angle relative to nvec - zero for plane wave
        const float   dof,  // the dof around foc over which to "flatten" the delay tabs
        const float* pvec,  // the point at which we are calculating delay tabs and apodization
        float* tau, float* apod // output numbers
    )
    {
        
        // calculate the magnitude of dx and its projection onto nvec
        float dxi, dxmag, dxproj;
        dxmag  = 0.0;
        dxproj = 0.0;
        for (int idim = 0; idim < ndim; ++idim)
        {
            dxi = pvec[idim] - foc[idim];
            dxmag  += dxi * dxi;
            dxproj += dxi * nvec[idim];
        }
        dxmag = sqrt(dxmag);

        // if synthetic focal point (diverging or converging waves)
        if (0.0 != ala)
        {
            if ((dof > 1E-9) && (abs(dxproj) <= dof/2)) {
                *tau = 2.0*(dxproj/dof)*(dxmag/c0) + t0;
                if (sqrt(abs(dxmag*dxmag - dxproj*dxproj)) <= dof * sin(ala) / 2.0) *apod = 1.0;
                else *apod = 0.0;
            }
            else {
                if (abs(dxproj) > 1E-9) *tau = (dxproj/abs(dxproj)) * (dxmag/c0) + t0;
                else *tau = t0;
                if ((abs(dxmag) > 1E-9) && (acos(abs(dxproj/dxmag)) > ala)) *apod = 0.0;
                else *apod = 1.0;
            }
        }

        // plane wave case
        else {
            *tau = dxproj/c0;
            *apod = 1.0;
        }
    }

    /**
     * calc_taurx_apodrx: calcualte the rx delay tabs and apodizations given recieve data structures
     * 
     * tau and apod are pointers to be filled with the correct values
     */
    void calc_taurx_apodrx(
        const int    ndim,  // 2 or 3 dimensions
        const float* orig,  // origin of receive element
        const float* nvec,  // normal vector of wave propagation
        const float    c0,  // assumed speed of sound in media
        const float   ala,  // the acceptance angle relative to nvec - zero for plane wave
        const float* pvec,  // the point at which we are calculating delay tabs and apodization
        float* tau, float* apod // output numbers
    )
    {
        // calculate the magnitude of dx and its projection onto nvec
        float dxi, dxmag, dxproj;
        dxmag  = 0.0;
        dxproj = 0.0;
        for (int idim = 0; idim < ndim; ++idim)
        {
            dxi = pvec[idim] - orig[idim];
            dxmag  += dxi * dxi;
            dxproj += dxi * nvec[idim];
        }
        dxmag = sqrt(dxmag);

        // calculate receive delay tabs and apodization
        *tau = dxmag/c0;
        if ((dxmag != 0.0) && (acos(abs(dxproj/dxmag)) > ala)) *apod = 0.0;
        else *apod = 1.0;
    }

    /**
     * ddas_bmode_synthetic_korder_cubic: beamform a DAS bmode using a synthetic point approach and korder cubic interpolation
     * 
     * RF channel data parameters:
     *   rfinfo: information about the rfdata
     *   rfdata: grid of rf data in the shape ntx by nrx by nt (stored in tInfo)
     * 
     * Transmit parameters:
     *   ovectx: origin of each transmision
     *   nvectx: normal vector of each of the transmit events
     *   t0tx: the timepoint at which the wave is at ovectx in each transmit event
     *   alatx: acceptance angle in radians relative to each nvectx
     * 
     * Receive parameters:
     *   ovecrx: origin of the receive points
     *   nvecrx: normal vector of each receive element
     *   alarx:  angular acceptance for each element relative to nvecrx
     * 
     * Interpolation parameters:
     *   k: number of points to include in interpolation kernel
     *   S: precomputed matrix to calculate the signal deriviatives
     * 
     * Field parameters:
     *   c0: the homogeneos speed of sound in the medium
     *   np: the number of recon points
     *   pvec: the location of each recon point
     *   pout: a vector length np, np*tx, np*rx, or np*tx*rx depending on output flag
     *   flag: int flag indicating if...
     *          - (0) not summing acros tx and rx
     *          - (1) summing across tx events only
     *          - (2) summing across rx events only
     *          - (3) summing across tx and rx events
     */
    __global__ 
    void das_bmode_synthetic_korder_cubic(
        const struct RFInfo rfinfo, const float* rfdata, 
        const float* ovectx, const float* nvectx, const float* t0tx, const float* alatx, const float* doftx, 
        const float* ovecrx, const float* nvecrx, const float* alarx,
        const float k, const float* S,
        const float c0, const unsigned int np, const float* pvec, float* pout,
        const int flag
    )
    {
        unsigned int tpb, bpg, tid, itx, irx, ip, ipout;
        float tautx, apodtx, taurx, apodrx;

        // get cuda step sizes
        tpb = ((unsigned int) (blockDim.x)) * ((unsigned int) (blockDim.y)) * ((unsigned int) (blockDim.z)); // threads per block

        // Unique thread ID
        tid  = threadIdx.x + threadIdx.y * blockDim.x;
        tid += threadIdx.z * blockDim.x * blockDim.y;
        tid += tpb * (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);

        if (tid >= (unsigned int)(rfinfo.ntx) * rfinfo.nrx * np) return;

        // calculate the transmit, recieve, and recon point indices for the thread we are working with
        itx = tid / (rfinfo.nrx * np);
        irx = (tid / np) % rfinfo.nrx;
        ip  = tid % np;

        calc_tautx_apodtx(
            rfinfo.ndim, 
            &ovectx[itx*rfinfo.ndim], 
            &nvectx[itx*rfinfo.ndim], 
            c0, t0tx[itx], 
            alatx[itx], 
            doftx[itx], 
            &pvec[ip*rfinfo.ndim],
            &tautx, &apodtx
        );

        calc_taurx_apodrx(
            rfinfo.ndim, 
            &ovecrx[irx*rfinfo.ndim], 
            &nvecrx[irx*rfinfo.ndim], 
            c0,
            alarx[irx], 
            &pvec[ip*rfinfo.ndim],
            &taurx, &apodrx
        );

        // If valid, add the beamformed and apodized value
        if (0 != apodtx * apodrx)
        {
            // calculate the index to save to based on the flag
            if      (0 == flag) ipout = ip + np*itx*irx;
            else if (1 == flag) ipout = ip + np*itx;
            else if (2 == flag) ipout = ip + np*irx;
            else                ipout = ip;

            // interpolate and add interpolated value to ipout
            atomicAdd(
                &pout[ipout], 
                apodtx * apodrx * korder_cubic_interp(
                    rfinfo.tInfo.x0, rfinfo.tInfo.dx, rfinfo.tInfo.nx, 
                    &rfdata[itx*rfinfo.nrx*rfinfo.tInfo.nx + irx*rfinfo.tInfo.nx], 
                    S, k,
                    tautx + taurx, 0.0
                )
            );
        } 
    }

    /**
     * ddas_bmode_synthetic_multi_interp: beamform a DAS bmode using a synthetic point approach and korder cubic interpolation
     * 
     * RF channel data parameters:
     *   rfinfo: information about the rfdata
     *   rfdata: grid of rf data in the shape ntx by nrx by nt (stored in tInfo)
     * 
     * Transmit parameters:
     *   ovectx: origin of each transmision
     *   nvectx: normal vector of each of the transmit events
     *   t0tx: the timepoint at which the wave is at ovectx in each transmit event
     *   alatx: acceptance angle in radians relative to each nvectx
     * 
     * Receive parameters:
     *   ovecrx: origin of the receive points
     *   nvecrx: normal vector of each receive element
     *   alarx:  angular acceptance for each element relative to nvecrx
     * 
     * Interpolation parameters:
     *   inter_flag: integer flag indicating what interpolation type to use
     *          - (0) nearest neighbor interpolation
     *          - (1) linear interpolation
     *          - (2) akima interpolation
     *          - (3) modified akima (makima) interpolation
     * 
     * Field parameters:
     *   c0: the homogeneos speed of sound in the medium
     *   np: the number of recon points
     *   pvec: the location of each recon point
     *   pout: a buffer of length np, np*tx, np*rx, or np*tx*rx depending on output flag
     *   flag: int flag indicating if...
     *          - (0) not summing acros tx and rx
     *          - (1) summing across tx events only
     *          - (2) summing across rx events only
     *          - (3) summing across tx and rx events
     */
    __global__ 
    void das_bmode_synthetic_multi_interp(
        const struct RFInfo rfinfo, const float* rfdata, 
        const float* ovectx, const float* nvectx, const float* t0tx, const float* alatx, const float* doftx, 
        const float* ovecrx, const float* nvecrx, const float* alarx,
        const int interp_flag,
        const float c0, const unsigned int np, const float* pvec, float* pout,
        const int flag
    )
    {
        unsigned int tpb, bpg, tid, itx, irx, ip, ipout;
        float tautx, apodtx, taurx, apodrx;

        // get cuda step sizes
        tpb = ((unsigned int) (blockDim.x)) * ((unsigned int) (blockDim.y)) * ((unsigned int) (blockDim.z)); // threads per block

        // Unique thread ID
        tid  = threadIdx.x + threadIdx.y * blockDim.x;
        tid += threadIdx.z * blockDim.x * blockDim.y;
        tid += tpb * (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);

        if (tid >= (unsigned int)(rfinfo.ntx) * rfinfo.nrx * np) return;

        // calculate the transmit, recieve, and recon point indices for the thread we are working with
        itx = tid / (rfinfo.nrx * np);
        irx = (tid / np) % rfinfo.nrx;
        ip  = tid % np;

        calc_tautx_apodtx(
            rfinfo.ndim, 
            &ovectx[itx*rfinfo.ndim], 
            &nvectx[itx*rfinfo.ndim], 
            c0, t0tx[itx], 
            alatx[itx], 
            doftx[itx], 
            &pvec[ip*rfinfo.ndim],
            &tautx, &apodtx
        );

        calc_taurx_apodrx(
            rfinfo.ndim, 
            &ovecrx[irx*rfinfo.ndim], 
            &nvecrx[irx*rfinfo.ndim], 
            c0,
            alarx[irx], 
            &pvec[ip*rfinfo.ndim],
            &taurx, &apodrx
        );

        // If valid, add the beamformed and apodized value
        if (0 != apodtx * apodrx)
        {
            // calculate the index to save to based on the flag
            if      (0 == flag) ipout = ip + np*itx*irx;
            else if (1 == flag) ipout = ip + np*itx;
            else if (2 == flag) ipout = ip + np*irx;
            else                ipout = ip;

            // interpolate and add interpolated value to ipout
            atomicAdd(
                &pout[ipout], 
                apodtx * apodrx * jumpTable[interp_flag](
                    rfinfo.tInfo.x0, rfinfo.tInfo.dx, rfinfo.tInfo.nx, 
                    &rfdata[itx*rfinfo.nrx*rfinfo.tInfo.nx + irx*rfinfo.tInfo.nx], 
                    tautx + taurx, 0.0
                )
            );
        } 
    }

    /**
     * das_bmode_tabbed_korder_cubic: beamform a DAS bmode using precomputed delay tabs
     * 
     * RF channel data parameters:
     *   rfinfo: information about the rfdata
     *   rfdata: grid of rf data in the shape ntx by nrx by nt (stored in tInfo)
     * 
     * Transmit parameters:
     *   tautx: the transmit delay tabs - in the shape ntx by np
     *   apodtx: the transmit apodizations - in the shape of ntx by np
     * 
     * Receive parameters:
     *   taurx: the recieve delay tabs - in the shape nrx by np
     *   apodrx: the recieve apodizations - in the shape of nrx by np
     * 
     * Interpolation parameters:
     *   k: number of points to include in interpolation kernel
     *   S: precomputed matrix to calculate the signal deriviatives
     * 
     * Field parameters:
     *   np: the number of recon points
     *   pvec: the location of each recon point
     *   pout: a buffer of length np, np*tx, np*rx, or np*tx*rx depending on output flag
     *   flag: int flag indicating if...
     *          - (0) not summing acros tx and rx
     *          - (1) summing across tx events only
     *          - (2) summing across rx events only
     *          - (3) summing across tx and rx events
     */
    __global__ 
    void das_bmode_tabbed_korder_cubic(
        const struct RFInfo rfinfo, const float* rfdata, 
        const float* tautx, const float* apodtx,
        const float* taurx, const float* apodrx,
        const int k, const float* S,
        const unsigned int np, float* pout, 
        const int flag
    )
    {
        unsigned int tpb, bpg, tid, itx, irx, ip, ipout;

        // get cuda step sizes
        tpb = ((unsigned int) (blockDim.x)) * ((unsigned int) (blockDim.y)) * ((unsigned int) (blockDim.z)); // threads per block

        // Unique thread ID
        tid  = threadIdx.x + threadIdx.y * blockDim.x;
        tid += threadIdx.z * blockDim.x * blockDim.y;
        tid += tpb * (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);

        if (tid >= (unsigned int)(rfinfo.ntx) * rfinfo.nrx * np) return;

        // calculate the transmit, recieve, and recon point indices for the thread we are working with
        itx = tid / (rfinfo.nrx * np);
        irx = (tid / np) % rfinfo.nrx;
        ip  = tid % np; 

        // If valid, add the beamformed and apodized value
        if (0 != apodtx[itx*np + ip] * apodrx[irx*np + ip])
        {
            // calculate the index to save to based on the flag
            if      (0 == flag) ipout = ip + np*irx + np*itx*rfinfo.nrx;
            else if (1 == flag) ipout = ip + np*irx;
            else if (2 == flag) ipout = ip + np*itx;
            else                ipout = ip;

            atomicAdd(
                &pout[ipout],        
                apodtx[itx*np + ip] * apodrx[irx*np + ip] * korder_cubic_interp(
                    rfinfo.tInfo.x0, rfinfo.tInfo.dx, rfinfo.tInfo.nx, 
                    &rfdata[itx*rfinfo.nrx*rfinfo.tInfo.nx + irx*rfinfo.tInfo.nx], 
                    S, k,
                    tautx[itx*np + ip] + taurx[irx*np + ip], 0.0
                ) 
            );
        } 
    }

    /**
     * das_bmode_tabbed_multi_interp: beamform a DAS bmode using precomputed delay tabs
     * 
     * RF channel data parameters:
     *   rfinfo: information about the rfdata
     *   rfdata: grid of rf data in the shape ntx by nrx by nt (stored in tInfo)
     * 
     * Transmit parameters:
     *   tautx: the transmit delay tabs - in the shape ntx by np
     *   apodtx: the transmit apodizations - in the shape of ntx by np
     * 
     * Receive parameters:
     *   taurx: the recieve delay tabs - in the shape nrx by np
     *   apodrx: the recieve apodizations - in the shape of nrx by np
     * 
     * Interpolation parameters:
     *   inter_flag: integer flag indicating what interpolation type to use
     *          - (0) nearest neighbor interpolation
     *          - (1) linear interpolation
     *          - (2) akima interpolation
     *          - (3) modified akima (makima) interpolation
     * 
     * Field parameters:
     *   np: the number of recon points
     *   pvec: the location of each recon point
     *   pout: a buffer of length np, np*tx, np*rx, or np*tx*rx depending on output flag
     *   flag: int flag indicating if...
     *          - (0) not summing acros tx and rx
     *          - (1) summing across tx events only
     *          - (2) summing across rx events only
     *          - (3) summing across tx and rx events
     */
    __global__ 
    void das_bmode_tabbed_multi_interp(
        const struct RFInfo rfinfo, const float* rfdata, 
        const float* tautx, const float* apodtx,
        const float* taurx, const float* apodrx,
        const int interp_flag,
        const unsigned int np, float* pout, 
        const int flag
    )
    {
        unsigned int tpb, bpg, tid, itx, irx, ip, ipout;

        // get cuda step sizes
        tpb = ((unsigned int) (blockDim.x)) * ((unsigned int) (blockDim.y)) * ((unsigned int) (blockDim.z)); // threads per block

        // Unique thread ID
        tid  = threadIdx.x + threadIdx.y * blockDim.x;
        tid += threadIdx.z * blockDim.x * blockDim.y;
        tid += tpb * (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);

        if (tid >= (unsigned int)(rfinfo.ntx) * rfinfo.nrx * np) return;

        // calculate the transmit, recieve, and recon point indices for the thread we are working with
        itx = tid / (rfinfo.nrx * np);
        irx = (tid / np) % rfinfo.nrx;
        ip  = tid % np; 

        // If valid, add the beamformed and apodized value
        if (0 != apodtx[itx*np + ip] * apodrx[irx*np + ip])
        {
            // calculate the index to save to based on the flag
            if      (0 == flag) ipout = ip + np*itx*irx;
            else if (1 == flag) ipout = ip + np*irx;
            else if (2 == flag) ipout = ip + np*itx;
            else                ipout = ip;

            atomicAdd(
                &pout[ipout],        
                apodtx[itx*np + ip] * apodrx[irx*np + ip] * jumpTable[interp_flag](
                    rfinfo.tInfo.x0, rfinfo.tInfo.dx, rfinfo.tInfo.nx, 
                    &rfdata[itx*rfinfo.nrx*rfinfo.tInfo.nx + irx*rfinfo.tInfo.nx], 
                    tautx[itx*np + ip] + taurx[irx*np + ip], 0.0
                ) 
            );
        } 
    }
}
