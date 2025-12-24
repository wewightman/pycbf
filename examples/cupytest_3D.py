import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import hilbert

from time import time
from timuscle.dataio import verasonics_loadtrackrf

with open("cupy_module.cu", mode='r') as fp: raw_module = fp.read()

module = cp.RawModule(code=raw_module)

das_bmode_cubic = module.get_function("das_bmode_cubic")

xInfo = np.dtype([('x0', np.float32),('dx', np.float32),('nx', np.int32)])
RFInfo = np.dtype([('ntx', np.int32),('nrx', np.int32),('ndim', np.int32),('tInfo', xInfo)])

datapath = "/fastrs/ultrasound/TIMuscle/InVivo/20230421_V001/acq_2_flex_45_musc_vl/RawData/"

rf, dims, rfpar = verasonics_loadtrackrf(datapath)

print(rf.shape, dims['steer_deg'])

xele = dims['xele_m']
steers = dims['steer_deg']
t = dims['t_sec']

c0 = 1540
r0 = 10

origtx = cp.ascontiguousarray(cp.array([[xele[-1] if steer <= 0 else xele[0] for steer in steers], np.zeros(steers.shape)], dtype=np.float32).T, dtype=cp.float32)

# calculate point source location to approximate plane waves
ovectx = cp.ascontiguousarray(-r0 * cp.array([np.sin(np.radians(steers)), np.cos(np.radians(steers))]).T, dtype=np.float32)

# calculate normal vector of point source
nvectx = cp.ascontiguousarray(cp.array([np.sin(np.radians(steers)), np.cos(np.radians(steers))]).T, dtype=np.float32)

t0tx = -cp.linalg.norm(origtx - ovectx, axis=-1)/c0

dof = cp.zeros(len(t0tx), dtype=np.float32)

# calculate acceptance angle for plane wave source approximated as a point
dxo = origtx - ovectx
alatx = cp.arccos(cp.abs(cp.sum(dxo * nvectx, axis=-1)) / cp.linalg.norm(dxo, axis=-1))

print(t0tx.dtype)

ovecrx = cp.ascontiguousarray(cp.array([xele, np.zeros(len(xele))]).T, dtype=np.float32)
nvecrx = cp.ascontiguousarray(cp.array([np.zeros(len(xele)), np.ones(len(xele))]).T, dtype=np.float32)
alarx = np.radians(30) * cp.ones(len(xele), dtype=np.float32)

rfinfo = np.zeros(1, dtype=RFInfo)
rfinfo['ntx'] = len(steers)
rfinfo['nrx'] = len(xele)
rfinfo['ndim'] = 2
rfinfo['tInfo']['x0'] = t[0]
rfinfo['tInfo']['dx'] = t[1] - t[0]
rfinfo['tInfo']['nx'] = len(t)

print(rfinfo)

xout = 1E-3*np.linspace(-22.5, 22.5, 501)
zout = 1E-3*np.arange(1, 60, 0.15/4)
Px, Pz = np.meshgrid(xout, zout, indexing='ij')
pvec = cp.ascontiguousarray(cp.array([Px, Pz]).transpose(2, 1, 0), dtype=np.float32)
print(rf.shape)
# exit()

allout = np.zeros((rf.shape[0], rf.shape[2], len(zout), len(xout)))

print("Starting beamforming")
tstart = time()
for ibatch in range(1):
    print("Copying data to GPU")
    t0copy = time()
    allrf = cp.ascontiguousarray(cp.array(rf[ibatch*6:(ibatch+1)*6,:,:,:,:]).transpose(0, 2, 3, 1, 4), dtype=np.float32)
    t1copy = time()
    print(f"  Copy time: {(t1copy-t0copy)*1E3} ms")
    pout = cp.zeros((6, rf.shape[2], len(zout), len(xout)), dtype=np.float32)
    for drot in range(6):
        irot = ibatch * 6 + drot
        t0rot = time()
        for iim in range(rf.shape[2]):
            params = (
                rfinfo,
                allrf[drot, iim],
                ovectx,
                nvectx,
                t0tx,
                alatx,
                dof,
                ovecrx,
                nvecrx,
                alarx,
                np.float32(c0),
                np.int32(pout[0,0].size),
                pvec,
                pout[drot,iim]
            )
        
            das_bmode_cubic((128,128,128), (1024,1,1), params)
            t1rot = time()
        print("  ", irot, " ", (t1rot - t0rot)*1E3, " ms")
    t0trans = time()
    allout[ibatch*6:(ibatch+1)*6,:,:,:] = cp.asnumpy(pout)
    t1trans = time()
    print("  tansfer out: ", (t1trans-t0trans)*1E3, " ms")
    del pout, allrf
tstop = time()
print(f"Done beamforming, {1E3*(tstop-tstart)} ms")

# env = np.abs(hilbert(pout.get(), axis=1))

# extent=1E3*np.array([xout[0], xout[-1], zout[-1], zout[0]])
# fig, axes = plt.subplots(6, 6, sharex=True, sharey=True)
# fig.set_size_inches(10, 12)
# fig.set_layout_engine("constrained")
# for irf in range(36):
#     logged = 20*np.log10(env[irf]/np.percentile(env[irf],99))
#     irow = irf//6
#     icol = irf%6
#     ax = axes[irow, icol]
#     ax.imshow(logged, vmin=-35, vmax=5, extent=extent, cmap='gray')
#     if irow == 5: ax.set_xlabel("Lateral [mm]")
#     if icol == 0: ax.set_ylabel("Axial [mm]")
# plt.savefig("das.png", dpi=300)

# print("oopa")