import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
from scipy.signal.windows import tukey
from scipy.signal import hilbert

from time import time
from timuscle.dataio import verasonics_loadtrackrf, putdictasHDF5

from pycbf import GPUBeamformer

datapath = "/fastrs/ultrasound/TIMuscle/InVivo/20230804_P007/acq_00_phantom_pf_30/RawData/"

rf, dims, rfpar = verasonics_loadtrackrf(datapath)

print(rf.shape, dims['steer_deg'])

xele = dims['xele_m']
steers = dims['steer_deg']
t = dims['t_sec']

c0 = 1540
r0 = 10

origtx = np.array([[xele[-1] if steer <= 0 else xele[0] for steer in steers], np.zeros(steers.shape)]).T

# calculate point source location to approximate plane waves
ovectx = -r0 * np.array([np.sin(np.radians(steers)), np.cos(np.radians(steers))]).T

# calculate normal vector of point source
nvectx = np.array([np.sin(np.radians(steers)), np.cos(np.radians(steers))]).T

t0tx = -np.linalg.norm(origtx - ovectx, axis=-1)/c0

dof = np.zeros(len(t0tx), dtype=np.float32)

# calculate acceptance angle for plane wave source approximated as a point
dxo = origtx - ovectx
alatx = np.arccos(np.abs(np.sum(dxo * nvectx, axis=-1)) / np.linalg.norm(dxo, axis=-1))

print(t0tx.dtype)

ovecrx = np.array([xele, np.zeros(len(xele))]).T
nvecrx = np.array([np.zeros(len(xele)), np.ones(len(xele))]).T
alarx = np.arctan2(1,2) * np.ones(len(xele), dtype=np.float32)


xout = 1E-3*np.linspace(-17.5, 17.5, 151)
zout = 1E-3*np.arange(1, 40, 0.15/4)
Px, Pz = np.meshgrid(xout, zout, indexing='ij')
pvec = np.array([Px.flatten(), Pz.flatten()]).T
print(rf.shape, pvec.shape)
# exit()

irot = 13+18

print("Starting beamforming")
tstart = time()

allrf = np.array(rf[irot,:,:,:,:]).transpose(1, 2, 0, 3)

pout = cp.zeros((rf.shape[2], rf.shape[3], len(zout), len(xout)), dtype=np.float32)

bmfrms = []
for istr in range(3):
    bmfrms.append(
        GPUBeamformer(
            ovectx = ovectx[istr:istr+1],
            nvectx = nvectx[istr:istr+1],
            t0tx   = t0tx[istr:istr+1],
            alatx  = alatx[istr:istr+1],
            doftx  = dof[istr:istr+1],
            ovecrx = ovecrx,
            nvecrx = nvecrx,
            alarx  = alarx,
            pnts   = pvec,
            c0 = 1540,
            t0 = t[0],
            dt = t[1] - t[0],
            nt = len(t),
            nthread = 512
        )
    )


t0rot = time()
for iim in range(rf.shape[2]):
    for istr in range(3):
        pout[iim,istr] = bmfrms[istr](
            allrf[iim, istr:istr+1], 
            out_as_numpy=False, 
            # buffer = pout[iim,istr].flatten()
        ).reshape(Px.shape[1], Px.shape[0])
t1rot = time()

print("  ", irot, " ", (t1rot - t0rot)*1E3, " ms")

print("Demodulating")
t0demod = time()
half = cp.fft.rfft(pout, axis=2)
half *= cp.array(tukey(half.shape[2], alpha=0.2))[None,None,:,None]
demod = cp.fft.ifft(cp.fft.ifftshift(half, axes=2), axis=2)
t1demod = time()
print("  Demodulation time: ", (t1demod-t0demod)*1E3, " ms")

t0trans = time()
allout = cp.asnumpy(pout)
demout = cp.asnumpy(demod)
t1trans = time()
print("  tansfer out: ", (t1trans-t0trans)*1E3, " ms")
del pout, allrf, demod

tstop = time()

print(f"Done beamforming, {1E3*(tstop-tstart)} ms")

putdictasHDF5("output.h5", {"bmodes":allout, "demod":demout, "lat":xout, "axial":zout})

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