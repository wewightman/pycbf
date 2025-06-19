import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

with open("cupy_module.cu", mode='r') as fp: raw_module = fp.read()

module = cp.RawModule(code=raw_module)
add_kernel = module.get_function('my_add')
linterp_kernel = module.get_function('my_linterp')
cubic_interp = module.get_function('my_cubeterp')

x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
y = cp.zeros((5, 5), dtype=cp.float32)
add_kernel((5,), (5,), (x1, x2, y))  # grid, block and arguments
print(y)

nxin = 32+1
scale = 16
nxout = scale*32+scale*2*16+1

yin = cp.array(np.sin(2*np.pi*np.arange(nxin)/4), dtype=np.float32)

xout = cp.linspace(-16, 32+16, nxout, dtype=np.float32)
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

yout_cube = cp.zeros(nxout, dtype=np.float32)
inputs = (
    np.float32(0),
    np.float32(1),
    np.int32(nxin),
    yin,
    xout,
    np.int32(nxout),
    yout_cube,
    np.float32(0)
)

cubic_interp((64,), (64,), inputs)

print(yin, yout_lin, yout_cube)

plt.figure()
plt.scatter(np.arange(nxin), yin.get(), label='Original', lw=1, color='black')
plt.plot(xout.get(), yout_lin.get(), label='Linear', lw=1)
plt.plot(xout.get(), yout_cube.get(), label='Cubic', lw=1)
plt.xlim(4,16)
plt.legend()

plt.savefig("test_interp.pdf")
