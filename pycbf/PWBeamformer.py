import ctypes
import os
import numpy as np
import ctypes
from multiprocessing import Pool, RawValue, RawArray
from pycbf import Beamformer, __BMFRM_PARAMS__, __BMFRM_DEBUG__

# python ctype wrappers for c engines
import pycbf.trig as trig

class PWBeamformer(Beamformer):
    """Right now, assumes all points are within y=0"""
    def __init__(self, c, fnum, points, alphas, trefs, refs, fs, tstart, nsamp:int, ncores=None, parallel=False):
        if __BMFRM_DEBUG__: print("Initializing a PWBeamformer...")
        Beamformer.__init__(self)

        def putsingles(data, dtype):
            res = []
            for datum in data:
                res.append(RawValue(dtype, datum))
            return res

        def putarrays(data, dtype):
            res = []
            for ind in range(data.shape[0]):
                res.append(RawArray(dtype, data.shape[1]))
                for indd in range(data.shape[1]):
                    res[ind][indd] = dtype(data[ind,indd])
            return res
        
        if __BMFRM_DEBUG__: print("  Formatting input parameters...")
        # copy singleton vectors to param structure
        params = {}
        params['npoints'] = points.shape[0]
        params['nacqs'] = refs.shape[0]
        params['c'] = c
        params['fnum'] = fnum
        params['nsamp'] = nsamp
        params['fs'] = fs
        params['tstart'] = tstart
        params['alphas'] = alphas
        params['refs'] = refs
        params['trefs'] = putsingles(trefs, ctypes.c_float)
        params['parallel'] = parallel

        if ncores is None:
            ncores = os.cpu_count()
            if ncores is None: ncores = int(1)
            params['ncores'] = ncores


        # copy large arrays to ctypes
        params['points'] = RawArray(ctypes.c_float, params['npoints']*3)
        _points = np.ascontiguousarray(points, dtype=ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for ind in range(params['npoints']*3):
            params['points'][ind] = _points[ind]

        # make a buffer of time index arrays
        params['tinds'] = []
        for ind in range(params['nacqs']):
            params['tinds'].append(RawArray(ctypes.c_int, params['npoints']))

        # make a intermediate buffer of tau arrays
        params['taus'] = []
        for ind in range(params['nacqs']):
            params['taus'].append(RawArray(ctypes.c_float, params['npoints']))
        
        # make a buffer of mask arrays
        params['masks'] = []
        for ind in range(params['nacqs']):
            params['masks'].append(RawArray(ctypes.c_int, params['npoints']))

        # make a buffer to store raw data
        params['datas'] = []
        for ind in range(params['nacqs']):
            params['datas'].append(RawArray(ctypes.c_float, params['nsamp']))

        # make a buffer to store raw data
        params['results'] = []
        for ind in range(params['nacqs']):
            params['results'].append(RawArray(ctypes.c_float, params['npoints']))

        # allocate final output array
        params['output'] = RawArray(ctypes.c_float, params['npoints'])

        __BMFRM_PARAMS__[self.id] = params

        self.__init_masks__()
        self.__init_tabs__()
    
    def __gen_tab__(self, ind):
        # load parameters
        params = __BMFRM_PARAMS__[self.id]
        npoints = ctypes.c_int(params['npoints'])
        nsamp = ctypes.c_int(params['nsamp'])
        ref = np.ascontiguousarray(params['refs'][ind,:], dtype=ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        tref = params['trefs'][ind]
        c = ctypes.c_float(params['c'])
        tstart = ctypes.c_float(params['tstart'])
        fs = ctypes.c_float(params['fs'])
        mask = params['masks'][ind]
        tau = params['taus'][ind]
        tind = params['tinds'][ind]
        alpha = float(params['alphas'][ind])
        norm = np.ascontiguousarray([np.sin(alpha), 0, np.cos(alpha)], dtype=ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # calculate time delays
        trig.fillarr(npoints, tau, ctypes.c_float(0))
        trig.pwtxengine(npoints, c, tref, ref, norm, params['points'], tau)
        trig.rxengine(npoints, c, ref, params['points'], tau)

        # calculate index to select
        trig.calcindices(npoints, nsamp, tstart, fs, tau, mask, tind)

    def __gen_mask__(self, ind):
        # load parameters
        params = __BMFRM_PARAMS__[self.id]
        npoints = ctypes.c_int(params['npoints'])
        ref = np.ascontiguousarray(params['refs'][ind,:], dtype=ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        fnum = ctypes.c_float(params['fnum'])
        alpha = float(params['alphas'][ind])
        mask = params['masks'][ind]
        focus = np.ascontiguousarray([np.sin(alpha), 0, np.cos(alpha)], dtype=ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        norm = np.ascontiguousarray([1, 0, 0], dtype=ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # generate mask
        trig.genmask3D(npoints, fnum, ctypes.c_int(1), fnum, ctypes.c_int(1), norm, focus, ref, params['points'], mask)

    def __init_tabs__(self):
        if __BMFRM_DEBUG__: print("    Generating Transmission Tabs")
        params = __BMFRM_PARAMS__[self.id]

        # if able to be parallelized do it, else...
        if params['parallel']:
            with Pool() as p:
                p.map(self.__gen_tab__, range(params['nacqs']))
        else:
            for inda in range(params['nacqs']):
                self.__gen_tab__(inda)
    
    def __init_masks__(self):
        if __BMFRM_DEBUG__: print("    Generating masks")
        params = __BMFRM_PARAMS__[self.id]

        # if able to be parallelized do it, else...
        if params['parallel']:
            with Pool() as p:
                p.map(self.__gen_mask__, range(params['nacqs']))
        else:
            for inda in range(params['nacqs']):
                self.__gen_mask__(inda)

    def __get_data__(self, ind):
        params = __BMFRM_PARAMS__[self.id]
        npoints = ctypes.c_int(params['npoints'])
        tind = params['tinds'][ind]
        data = params['datas'][ind]
        result = params['results'][ind]

        trig.selectdata(npoints, tind, data, result)

    def __copy_data_2_buffer__(self, data):
        params = __BMFRM_PARAMS__[self.id]
        nacqs = params['nacqs']
        nsamp = params['nsamp']
        datas = params['datas']
        norig = ctypes.c_int(nacqs * nsamp)
        nsub = ctypes.c_int(nsamp)
        for inda in range(nacqs):
            trig.copysubvec(norig, nsub, ctypes.c_int(inda), data, datas[inda])


    def __call__(self, data):
        params = __BMFRM_PARAMS__[self.id]
        nacqs = params['nacqs']
        npoints = params['npoints']

        if __BMFRM_DEBUG__: print("Beamforming...\n  Copying data")
        self.__copy_data_2_buffer__(data)

        # send data collection to parallelization pool (eg delay)
        if __BMFRM_DEBUG__: print("  Starting pool...")
        if params['parallel']:
            with Pool() as p:
                p.map(self.__get_data__, range(nacqs))
        else:
            for inda in range(nacqs):
                self.__get_data__(inda)
        
        # sum up all output vectors and clear from memory(eg and sum)
        if __BMFRM_DEBUG__: print("  Finished pool. Summing...")
        results = params['results']
        output = params['output']
        trig.fillarr(npoints, output, ctypes.c_float(0))
        for idr in range(len(results)):
            trig.sumvecs(npoints, output, results[idr], ctypes.c_float(0), output)
        
        if __BMFRM_DEBUG__: print("  Converting to output array")
        return np.array([output[ind] for ind in range(npoints)])
        