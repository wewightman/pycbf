extern "C" {

    /**
     * cube_interp: subeic interpolation assuming regular spacing
     */
    float cube_interp(
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
     * das_bmode_cubic: beamform a DAS bmode 
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
     * Field parameters:
     *   c0: the homogeneos speed of sound in the medium
     *   np: the number of recon points
     *   pvec: the location of each recon point
     *   pout: a vector length p for the output bmode
     */
    __global__ 
    void das_bmode_cubic(
        const struct RFInfo rfinfo, const float* rfdata, 
        const float* ovectx, const float* nvectx, const float* t0tx, const float* alatx, const float* doftx, 
        const float* ovecrx, const float* nvecrx, const float* alarx,
        const float c0, const int np, const float* pvec, float* pout
    )
    {
        int tpb, bpg, tid, itx, irx, ip;
        float tautx, apodtx, taurx, apodrx;

        // get cuda step sizes
        tpb = blockDim.x * blockDim.y * blockDim.z; // threads per block

        // Unique thread ID
        tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        tid += tpb * blockIdx.x + tpb * blockIdx.y * gridDim.x + tpb * blockIdx.z * gridDim.x * gridDim.y;

        if (tid >= rfinfo.ntx * rfinfo.nrx * np) return;

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
            atomicAdd(
                &pout[ip], 
                apodtx * apodrx * cube_interp(rfinfo.tInfo.x0, rfinfo.tInfo.dx, rfinfo.tInfo.nx, &rfdata[itx*rfinfo.nrx*rfinfo.tInfo.nx + irx*rfinfo.tInfo.nx], tautx + taurx, 0.0)
            );
        } 
    }

    /**
     * das_bmode_rxseparate_cubic: beamform data keeping RX data separate
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
     * Field parameters:
     *   c0: the homogeneos speed of sound in the medium
     *   np: the number of recon points
     *   pvec: the location of each recon point
     *   pout: a vector length p x rx for the output bmode
     */
    __global__ 
    void das_bmode_rxseparate_cubic(
        const struct RFInfo rfinfo, const float* rfdata, 
        const float* ovectx, const float* nvectx, const float* t0tx, const float* alatx, const float* doftx, 
        const float* ovecrx, const float* nvecrx, const float* alarx,
        const float c0, const int np, const float* pvec, float* pout
    )
    {
        int tpb, bpg, tid, itx, irx, ip;
        float tautx, apodtx, taurx, apodrx;

        // get cuda step sizes
        tpb = blockDim.x * blockDim.y * blockDim.z; // threads per block

        // Unique thread ID
        tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        tid += tpb * blockIdx.x + tpb * blockIdx.y * gridDim.x + tpb * blockIdx.z * gridDim.x * gridDim.y;

        if (tid >= rfinfo.ntx * rfinfo.nrx * np) return;

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

        if (0 != apodtx * apodrx)
        {
            atomicAdd(
                &pout[irx * np + ip], 
                apodtx * apodrx * cube_interp(rfinfo.tInfo.x0, rfinfo.tInfo.dx, rfinfo.tInfo.nx, &rfdata[itx*rfinfo.nrx*rfinfo.tInfo.nx + irx*rfinfo.tInfo.nx], tautx + taurx, 0.0)
            );
        }
    }
}
