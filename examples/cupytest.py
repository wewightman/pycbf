import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator

with open("cupy_module.cu", mode='r') as fp: raw_module = fp.read()

module = cp.RawModule(code=raw_module)
linterp_kernel = module.get_function('my_linterp')
cubic_interp = module.get_function('my_cubeterp')

nxin = 32+1
scale = 16
nxout = scale*16+scale*2*16+1

yin = cp.array(np.sin(2*np.pi*np.arange(nxin)/4), dtype=np.float32)
yin[:8] = 0
yin[-8:] = 0

xout = cp.linspace(-8, 16+16+8, nxout, dtype=np.float32)
yout_lin = cp.zeros(nxout, dtype=np.float32)
inputs = (
    np.float32(0),
    np.float32(1),
    np.int32(nxin),
    yin,
    xout,
    np.int32(nxout),
    yout_lin,
    np.float32(0)
)

linterp_kernel((32,), (32,), inputs)

xInfo = np.dtype([('x0', np.float32),('dx', np.float32),('nx', np.int32)])
xinfo = np.zeros(1, dtype=xInfo)#.view(xInfo)
print(xinfo)
xinfo['x0'] = 0
xinfo['dx'] = 1
xinfo['nx'] = nxin
print(xinfo)

yout_cube = cp.zeros(nxout, dtype=np.float32)
inputs = (
    xinfo,
    yin,
    xout,
    np.int32(nxout),
    yout_cube,
    np.float32(0)
)

cubic_interp((64,), (64,), inputs)

fint = Akima1DInterpolator(np.arange(nxin), yin.get(), method='makima')
yout_scipy = fint(xout.get())

plt.figure()
plt.scatter(np.arange(nxin), yin.get(), label='Original', lw=1, color='black')
plt.plot(xout.get(), yout_lin.get(), label='Linear', lw=1)
plt.plot(xout.get(), yout_cube.get(), label='Cubic', lw=1)
plt.legend()
plt.xlim(-2, 34)
plt.savefig("test_interp.pdf")

fig, axes = plt.subplots(2, 1, sharex=True)
ax = axes[0]
ax.scatter(np.arange(nxin), yin.get(), label='Original', lw=1, color='black')
ax.plot(xout.get(), yout_cube.get(), label='Custom mAkima', lw=1)
ax.plot(xout.get(), yout_scipy, label='Scipy mAkima', lw=1)
ax.legend()
ax.set_xlim(-2, 34)

ax = axes[1]
ax.plot(xout.get(), yout_cube.get() - yout_scipy, label='Custom - Scipy', lw=1)
ax.legend()
plt.savefig("test_interp_scipy.pdf")

RFInfo = np.dtype([('ntx', np.int32),('nrx', np.int32),('np', np.int32),('ndim', np.int32),('tInfo', xInfo)])