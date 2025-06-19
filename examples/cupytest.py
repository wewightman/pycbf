import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

add_kernel = cp.RawKernel(r'''
extern "C" __global__
void my_add(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + 2*x2[tid]-10;
}
''', 'my_add')
x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
y = cp.zeros((5, 5), dtype=cp.float32)
add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
print(y)

linterp_kernel = cp.RawKernel(r'''
extern "C" __global__
void my_linterp(const float x0, const float dx, const int nx, const float* y, const float* xout, const int nxout, float* yout, float fill) {
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
''', 'my_linterp')

nxin = 32+1
nxout = 2*32+4*16+1

yin = cp.array(np.sin(2*np.pi*np.arange(nxin)/8), dtype=np.float32)

xout_lin = cp.linspace(-16, 32+16, nxout, dtype=np.float32)
yout_lin = cp.zeros(nxout, dtype=np.float32)
inputs = (
    np.float32(0),
    np.float32(1),
    np.int32(nxin),
    yin,
    xout_lin,
    np.int32(nxout),
    yout_lin,
    np.float32(0)
)

linterp_kernel((32,), (32,), inputs)

print(yin, yout_lin)

plt.figure()
plt.plot(np.arange(nxin), yin.get(), label='Original')
plt.plot(xout_lin.get(), yout_lin.get(), label='Linear', linestyle='--')
plt.legend()
plt.show()
