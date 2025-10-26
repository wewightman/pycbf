extern "C" {

    /**
     * calc_dmas: This kernel calculate a DMAS image, multiplying and summing along the correlation axis with the option to sum the lags
     */
    __global__
    void calc_dmas(
        const float* imsep,     // beamformed images before summing on the correlation axis (nsep by np)
        const int    nsep,      // number of separated images on the correlation axis
        const int    np,        // number of points in the image
        const int*   lags,      // an integer array specifying the lag indices to sum over
        const int    nlag,      // the number of lags in the lag array
        const int    sumtype,   // a flag indicating signed-square-root DMAS (0) or power DMAS (1)
        const int    sumlags,   // a flag indicating whether to sub across the lags (1) or not (0)
        float*       imout      // the DMAS image - size np or nlag by np depending on sumlags
    )
    {
        int tpb, ilag, isep, ip, ipout;
        float product;

        // get cuda step sizes
        tpb = blockDim.x * blockDim.y * blockDim.z; // threads per block

        // Unique thread ID
        tid  = threadIdx.x + threadIdx.y * blockDim.x;
        tid += threadIdx.z * blockDim.x * blockDim.y;
        tid += tpb * (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);

        // exit early if out of bounds
        if (tid >= nlag * nsep * np) return;

        // calcualte the process indices
        ilag = tid / (nsep * np);
        isep = (tid / np) % nsep;
        ip = tid % np;

        // exit if lag is out of bounds
        if (isep + ilag >= nsep) return;

        // determine the output buffer index
        if (sumlags) ipout = ip;
        else ipout = ip + ilag*np;

        // multiply th lag pairs
        product = imsep[ip + np*isep] * imsep[ip + np*(isep + ilag)];

        // process the multiplication and sum
        if      (sumtype == 0) atomicAdd(&imout[ipout], sign(product)*abs(product));
        else if (sumtype == 1) atomicAdd(&imout[ipout], product);
    }
}