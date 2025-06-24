import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator

with open("cupy_module.cu", mode='r') as fp: raw_module = fp.read()

module = cp.RawModule(code=raw_module)

das_bmode_cubic = module.get_function("das_bmode_cubic")

xInfo = np.dtype([('x0', np.float32),('dx', np.float32),('nx', np.int32)])
RFInfo = np.dtype([('ntx', np.int32),('nrx', np.int32),('ndim', np.int32),('tInfo', xInfo)])

rfinfo = np.zeros(1, dtype=RFInfo)
rfinfo['ntx'] = 1.0
rfinfo['nrx'] = 1.0
rfinfo['ndim'] = 2
rfinfo['tInfo']['x0'] = 0.0
rfinfo['tInfo']['dx'] = 1E-6
rfinfo['tInfo']['nx'] = 1024

als = np.radians(0)
c0 = 1540

xout = 1E-3*np.linspace(-40, 40, 401)
zout = 1E-3*np.linspace(1, 81, 401)
Px, Pz = np.meshgrid(xout, zout, indexing='ij')
pvec = cp.ascontiguousarray(cp.array([Px, Pz]).transpose(2, 1, 0), dtype=np.float32)
pout = cp.zeros(pvec.shape[:-1], dtype=np.float32)

params = (
    rfinfo,
    cp.zeros((1, 1, 1024), dtype=np.float32),
    cp.array([-5E-3, -40E-3], dtype=np.float32),
    cp.array([np.sin(als), np.cos(als)], dtype=np.float32),
    cp.array([35E-3/c0], dtype=np.float32),
    cp.array([np.radians(10)], dtype=np.float32),
    cp.array([1E-3], dtype=np.float32),
    cp.array([-10E-3, 0], dtype=np.float32),
    cp.array([0, 1], dtype=np.float32),
    cp.array([np.radians(45)], dtype=np.float32),
    np.float32(c0),
    np.int32(pout.size),
    pvec,
    pout
)

das_bmode_cubic((64,64,64), (1024,1,1), params)

plt.figure()
plt.imshow(pout.get())
plt.colorbar()
plt.savefig("das.pdf")