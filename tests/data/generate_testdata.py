"""This script generates all test data using the beamformer at a reference date. 

Changes to algorithms (I am mainly reffering to the interpolation algorithms that underpin the beamformers) will lead to some tests failing. It will be on the maintainer to validate if those differences are are correct chagnes that need to be added (eg if an error in the existsing interpolation code is detected).

This doc string indicates the dates that the test data was genrated on, with the most recent additions at the top.

Dates, user, and reason for regenerating test data:
- 2026/02/04, Wren Wightman, preliminary generation of test data
"""
from pytest import mark

__PLOT_RESULTS_TEST_DATA__ = False

@mark.skip
def make_simple_interpolator_data():
    """Makes example data and inteprolates it using different beamforming configurations"""
    import numpy as np
    import matplotlib.pyplot as plt
    from pycbf.dataio.__hdf5_engines__ import putdictasHDF5
    from pycbf.cpu.__core_bf_classes__ import TabbedBeamformer
    fc = 5E6
    tin = np.arange(-3E-6, 3E-6, 1/(4*fc))
    tout = np.arange(-5E-6, 5E-6, 1/(128*fc))

    raw_sig = tin * np.sin(2*np.pi*fc*tin) * np.exp(-(tin*fc/8)**2)

    interp_keys = dict()
    interp_keys['cubic'] = dict(kind='cubic')
    interp_keys['nearest'] = dict(kind='nearest')
    for usf in [1, 2, 4, 8, 16, 32]:
        interp_keys[f'nearest_usf{usf:02d}'] = dict(kind='nearest', usf=usf)

    tautx = np.zeros(len(tout))[None,:]
    apodtx = np.ones(len(tout))[None,:]
    apodrx = np.ones(len(tout))[None,:]
    results = dict()

    for key, interp in interp_keys.items():
        bf = TabbedBeamformer(
            tautx=tautx,
            taurx=tout[None,:],
            apodtx=apodtx,
            apodrx=apodrx,
            interp=interp,
            t0 = tin[0],
            dt=tin[1]-tin[0],
            nt=len(tin)
        )
        results[key] = bf(raw_sig[None,None,:])

    if __PLOT_RESULTS_TEST_DATA__:
        plt.figure()
        plt.plot(1E6*tin, raw_sig)
        for key in interp_keys.keys(): plt.plot(1E6*tout, results[key])
        plt.show()

    putdictasHDF5(
        "interpolator_groundtruth_data.h5",
        dict(
            tin=tin,
            tout=tout,
            sig_orig = raw_sig,
            interp_keys = interp_keys,
            sig_interp = results
        )
    )

@mark.skip
def make_mask_data():
    """This function generates mask data to compare against for a few special cases"""
    raise Exception("make_mask_data has not been implemented yet")

if __name__ == '__main__':
    make_simple_interpolator_data()
    make_mask_data()
    
    