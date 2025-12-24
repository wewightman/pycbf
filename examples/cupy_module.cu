extern "C" {
    __global__ void my_add(const float* x1, const float* x2, float* y) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        y[tid] = x1[tid] + 2*x2[tid]-10;
    }

    __global__ void my_linterp(const float x0, const float dx, const int nx, const float* y, const float* xout, const int nxout, float* yout, float fill) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= nxout) return;
                                
        float xn = x0 + dx * (nx-1);
        float xo = xout[tid];
        
        if (xo == xn) {
            yout[tid] = y[nx-1];
            return;
        } else if ((xo < x0) || (xo > xn)) {
            yout[tid] = fill;
            return;
        }
                                
        int ixo = (int) ((xo - x0)/dx);
        float xi = x0 + ixo * dx;
        float delta = (xo - xi)/dx;
        yout[tid] = (1-delta) * y[ixo] + delta * y[ixo+1];
    }

    __global__ void my_cubeterp(const float x0, const float dx, const int nx, const float* y, const float* xout, const int nxout, float* yout, float fill) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= nxout) return;
                                
        float xn = x0 + dx * (nx-1);
        float xo = xout[tid];
        
        if (xo == xn) {
            yout[tid] = y[nx-1];
            return;
        } else if ((xo < x0) || (xo > xn)) {
            yout[tid] = fill;
            return;
        }
                                
        int ixo = (int) ((xo - x0)/dx);

        float mm2, mm1, mp0, mp1, mp2, w0, w1, sp0, sp1, a, b, c, d, delta;

        // boundary conditions (bc)- first point
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

            sp0 = (mp0 + mp1)/2;

            w0  = abs(mp2 - mp1) + abs(mp2 + mp1)/2;
            w1  = abs(mp0 - mm1) + abs(mp0 + mp1)/2;
            sp1 = (w0 * mp0 + w1 * mp1) / (w0 + w1);

        } 
        // bc - third to last point
        else if (ixo == nx-3) {
            mm2 = y[ixo+-1] - y[ixo-2];
            mm1 = y[ixo+0] - y[ixo-1];
            mp0 = y[ixo+1] - y[ixo-0];
            mp1 = y[ixo+2] - y[ixo+1];

            w0  = abs(mp1 - mp0) + abs(mp1 + mp0)/2;
            w1  = abs(mm1 - mm2) + abs(mm1 + mm2)/2;
            sp0 = (w0 * mm1 + w1*mp0) / (w0 + w1);

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

            w0  = abs(mp1 - mp0) + abs(mp1 + mp0)/2;
            w1  = abs(mm1 - mm2) + abs(mm1 + mm2)/2;
            sp0 = (w0 * mm1 + w1*mp0) / (w0 + w1);

            w0  = abs(mp2 - mp1) + abs(mp2 + mp1)/2;
            w1  = abs(mp0 - mm1) + abs(mp0 + mm1)/2;
            sp1 = (w0 * mp0 + w1*mp1) / (w0 + w1);
        }

        a = y[ixo];
        b = sp0;
        c = (3*mp0 - 2*sp0 - sp1)/dx;
        d = (sp0 + sp1 - 2*mp0)/(dx*dx);

        delta = xout[tid] - (x0 + dx * ixo);

        yout[tid] = a + b * delta + c * delta * delta + d * delta * delta * delta;
    }
}
