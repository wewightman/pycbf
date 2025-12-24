from dataclasses import dataclass, field
from pycbf.__bf_base_classes__ import Synthetic, Tabbed, Parallelized, BeamformerException, __BMFRM_PARAMS__

from numpy import ndarray as npNDArray
from cupy  import ndarray as cpNDArray

@dataclass(kw_only=True)
class SyntheticDAS(Synthetic, Parallelized):
    t0 : float = field(init=True)
    dt : float = field(init=True)
    nt :   int = field(init=True)
    c0 : float = field(init=True)
    nthread : int = 512

    def __post_init__(self):
        Parallelized.__post_init__(self)
        Synthetic.__post_init__(self)

        from cupy import array, ascontiguousarray, float32
        from pycbf.gpu.__engine__ import RFInfo
        import numpy as np
        params = dict()

        # copy TX parameters into shared GPU memory
        params['ovectx'] = ascontiguousarray(array(self.ovectx), dtype=float32)
        params['nvectx'] = ascontiguousarray(array(self.nvectx), dtype=float32)
        params[ 'doftx'] = ascontiguousarray(array(self. doftx), dtype=float32)
        params[ 'alatx'] = ascontiguousarray(array(self. alatx), dtype=float32)
        params[  't0tx'] = ascontiguousarray(array(self.  t0tx), dtype=float32)

        # copy RX parameters into shared GPU memory
        params['ovecrx'] = ascontiguousarray(array(self.ovecrx), dtype=float32)
        params['nvecrx'] = ascontiguousarray(array(self.nvecrx), dtype=float32)
        params[ 'alarx'] = ascontiguousarray(array(self. alarx), dtype=float32)

        # copy output pnts into shared GPU memory
        params[  'pnts'] = ascontiguousarray(array(self.  pnts), dtype=float32)

        # dimensions
        params[  'ntx'] = self.ntx
        params[  'nrx'] = self.nrx
        params[  'nop'] = self.nop
        params['ndimp'] = self.ndimp

        # make struct describing data
        params['rfinfo'] = np.zeros(1, dtype=RFInfo)
        params['rfinfo']['ntx']  = self.ntx
        params['rfinfo']['nrx']  = self.nrx
        params['rfinfo']['ndim'] = self.ndimp
        params['rfinfo']['tInfo']['x0'] = self.t0
        params['rfinfo']['tInfo']['dx'] = self.dt
        params['rfinfo']['tInfo']['nx'] = self.nt

        __BMFRM_PARAMS__[self.id] = params

        

    def __call__(self, txrxt : cpNDArray | npNDArray, out_as_numpy : bool = True, buffer : cpNDArray | None = None) -> cpNDArray | npNDArray:
        """Beamform txrxt tensor """
        import cupy as cp
        import numpy as np
        from pycbf.gpu.__engine__ import das_bmode_cubic_synthetic as gpu_kernel

        if isinstance(txrxt, npNDArray):
            txrxt = cp.ascontiguousarray(cp.array(txrxt), dtype=np.float32)
        elif isinstance(txrxt, cpNDArray):
            if (txrxt.dtype != np.float32) or (txrxt.dtype != cp.float32):
                raise BeamformerException("Cupy array dtype must be either cupy or numpy float 32")
        else:
            raise BeamformerException("txrxt must be an instance of either a cupy or numpy ndarray but was ", type(txrxt))
        
        if isinstance(txrxt, cpNDArray):
            if not txrxt.flags['C_CONTIGUOUS']:
                txrxt = cp.ascontiguousarray(txrxt, dtype=np.float32)

        if buffer is None: pout = cp.zeros(self.nop, dtype=np.float32)
        else: raise Exception("Something is wrong with input buffers") #pout = buffer

        bf_params = __BMFRM_PARAMS__[self.id]
        routine_params = (
            bf_params['rfinfo'],
            txrxt,
            bf_params['ovectx'],
            bf_params['nvectx'],
            bf_params[  't0tx'],
            bf_params[ 'alatx'],
            bf_params[ 'doftx'],
            bf_params['ovecrx'],
            bf_params['nvecrx'],
            bf_params[ 'alarx'],
            np.float32(self.c0),
            np.int32(self.nop),
            bf_params[  'pnts'],
            pout
        )

        nblock = np.int32(np.ceil(self.ntx * self.nrx * self.nop / self.nthread))
        gpu_kernel((nblock,1,1), (self.nthread,1,1), routine_params)

        if out_as_numpy: return cp.asnumpy(pout)
        else: return pout

    def __del__(self):
        params = __BMFRM_PARAMS__[self.id]

        # delete globally stored data if it exists
        for key in ['ovectx', 'nvectx', 'doftx', 'alatx', 't0tx', 'ovecrx', 'nvecrx', 'alarx', 'pnts']:
            if key in params.keys(): del params[key]

        del __BMFRM_PARAMS__[self.id]


@dataclass(kw_only=True)
class SyntheticDAS_RxSeparate(SyntheticDAS):

    def __call__(self, txrxt : cpNDArray | npNDArray, out_as_numpy : bool = True, buffer : cpNDArray | None = None) -> cpNDArray | npNDArray:
        """Beamform txrxt tensor """
        import cupy as cp
        import numpy as np
        from pycbf.gpu.__engine__ import das_bmode_rxseparate_cubic_synthetic as gpu_kernel

        if isinstance(txrxt, npNDArray):
            txrxt = cp.ascontiguousarray(cp.array(txrxt), dtype=np.float32)
        elif isinstance(txrxt, cpNDArray):
            if (txrxt.dtype != np.float32) or (txrxt.dtype != cp.float32):
                raise BeamformerException("Cupy array dtype must be either cupy or numpy float 32")
        else:
            raise BeamformerException("txrxt must be an instance of either a cupy or numpy ndarray but was ", type(txrxt))
        
        if isinstance(txrxt, cpNDArray):
            if not txrxt.flags['C_CONTIGUOUS']:
                txrxt = cp.ascontiguousarray(txrxt, dtype=np.float32)

        if buffer is None: pout = cp.zeros(self.nop*self.nrx, dtype=np.float32)
        else: raise Exception("Something is wrong with input buffers") #pout = buffer

        bf_params = __BMFRM_PARAMS__[self.id]
        routine_params = (
            bf_params['rfinfo'],
            txrxt,
            bf_params['ovectx'],
            bf_params['nvectx'],
            bf_params[  't0tx'],
            bf_params[ 'alatx'],
            bf_params[ 'doftx'],
            bf_params['ovecrx'],
            bf_params['nvecrx'],
            bf_params[ 'alarx'],
            np.float32(self.c0),
            np.int32(self.nop),
            bf_params[  'pnts'],
            pout
        )

        nblock = np.int32(np.ceil(self.ntx * self.nrx * self.nop / self.nthread))
        gpu_kernel((nblock,1,1), (self.nthread,1,1), routine_params)

        if out_as_numpy: return cp.asnumpy(pout)
        else: return pout

    def __del__(self):
        params = __BMFRM_PARAMS__[self.id]

        # delete globally stored data if it exists
        for key in ['ovectx', 'nvectx', 'doftx', 'alatx', 't0tx', 'ovecrx', 'nvecrx', 'alarx', 'pnts']:
            if key in params.keys(): del params[key]

        del __BMFRM_PARAMS__[self.id]

@dataclass(kw_only=True)
class TabbedDAS(Tabbed, Parallelized):
    t0 : float = field(init=True)
    dt : float = field(init=True)
    nt :   int = field(init=True)
    thresh : float = 1E-2
    nthread : int = 512

    def __post_init__(self):
        Parallelized.__post_init__(self)
        Tabbed.__post_init__(self)

        from cupy import array, ascontiguousarray, float32
        from pycbf.gpu.__engine__ import RFInfo
        import numpy as np

        # Access the global shared buffer
        params = dict()

        # copy tx/rx/output point dimensions
        params['nop']    = self.nop
        params['thresh'] = self.thresh

        # make struct describing data
        params['rfinfo'] = np.zeros(1, dtype=RFInfo)
        params['rfinfo']['ntx']  = self.ntx
        params['rfinfo']['nrx']  = self.nrx
        params['rfinfo']['ndim'] = 0
        params['rfinfo']['tInfo']['x0'] = self.t0
        params['rfinfo']['tInfo']['dx'] = self.dt
        params['rfinfo']['tInfo']['nx'] = self.nt

        # Copy beamforming tabs to CPU memory
        params['tautx' ] = ascontiguousarray(array(self.tautx ), dtype=float32)
        params['taurx' ] = ascontiguousarray(array(self.taurx ), dtype=float32)
        params['apodtx'] = ascontiguousarray(array(self.apodtx), dtype=float32)
        params['apodrx'] = ascontiguousarray(array(self.apodrx), dtype=float32)

        __BMFRM_PARAMS__[self.id] = params

    def __call__(self, txrxt : cpNDArray | npNDArray, out_as_numpy : bool = True, buffer : cpNDArray | None = None) -> cpNDArray | npNDArray:

        import cupy as cp
        import numpy as np
        from pycbf.gpu.__engine__ import das_bmode_cubic_tabbed as gpu_kernel

        if isinstance(txrxt, npNDArray):
            txrxt = cp.ascontiguousarray(cp.array(txrxt), dtype=np.float32)
        elif isinstance(txrxt, cpNDArray):
            if (txrxt.dtype != np.float32) or (txrxt.dtype != cp.float32):
                raise BeamformerException("Cupy array dtype must be either cupy or numpy float 32")
        else:
            raise BeamformerException("txrxt must be an instance of either a cupy or numpy ndarray but was ", type(txrxt))
        
        if isinstance(txrxt, cpNDArray):
            if not txrxt.flags['C_CONTIGUOUS']:
                txrxt = cp.ascontiguousarray(txrxt, dtype=np.float32)

        if buffer is None: pout = cp.zeros(self.nop, dtype=np.float32)
        else: raise Exception("Something is wrong with input buffers") #pout = buffer

        k = 16
        S = cp.ascontiguousarray(cp.array(__make_S_by_k__(k)), dtype=np.float32)

        bf_params = __BMFRM_PARAMS__[self.id]
        routine_params = (
            bf_params['rfinfo'],
            txrxt,
            bf_params['tautx'],
            bf_params['apodtx'],
            bf_params['taurx'],
            bf_params['apodrx'],
            np.int32(k), S,
            np.int32(self.nop),
            pout
        )

        nblock = np.int32(np.ceil(self.ntx * self.nrx * self.nop / self.nthread))

        gpu_kernel((nblock,1,1), (self.nthread,1,1), routine_params)

        if out_as_numpy: return cp.asnumpy(pout)
        else: return pout

    def __del__(self):
        params = __BMFRM_PARAMS__[self.id]

        # delete globally stored data if it exists
        for key in ['tautx', 'taurx', 'apodtx', 'apodrx']:
            if key in params.keys(): del params[key]

        del __BMFRM_PARAMS__[self.id]

@dataclass(kw_only=True)
class TabbedDAS_RxSeparate(TabbedDAS):

    def __call__(self, txrxt : cpNDArray | npNDArray, out_as_numpy : bool = True, buffer : cpNDArray | None = None) -> cpNDArray | npNDArray:

        import cupy as cp
        import numpy as np
        from pycbf.gpu.__engine__ import das_bmode_rxseparate_cubic_tabbed as gpu_kernel

        if isinstance(txrxt, npNDArray):
            txrxt = cp.ascontiguousarray(cp.array(txrxt), dtype=np.float32)
        elif isinstance(txrxt, cpNDArray):
            if (txrxt.dtype != np.float32) or (txrxt.dtype != cp.float32):
                raise BeamformerException("Cupy array dtype must be either cupy or numpy float 32")
        else:
            raise BeamformerException("txrxt must be an instance of either a cupy or numpy ndarray but was ", type(txrxt))
        
        if isinstance(txrxt, cpNDArray):
            if not txrxt.flags['C_CONTIGUOUS']:
                txrxt = cp.ascontiguousarray(txrxt, dtype=np.float32)

        if buffer is None: pout = cp.zeros(self.nrx*self.nop, dtype=np.float32)
        else: raise Exception("Something is wrong with input buffers") #pout = buffer

        k = 16
        S = cp.ascontiguousarray(cp.array(__make_S_by_k__(k)), dtype=np.float32)

        bf_params = __BMFRM_PARAMS__[self.id]
        routine_params = (
            bf_params['rfinfo'],
            txrxt,
            bf_params['tautx'],
            bf_params['apodtx'],
            bf_params['taurx'],
            bf_params['apodrx'],
            np.int32(k), S,
            np.int32(self.nop),
            pout
        )

        nblock = np.int32(np.ceil(self.ntx * self.nrx * self.nop / self.nthread))

        gpu_kernel((nblock,1,1), (self.nthread,1,1), routine_params)

        if out_as_numpy: return cp.asnumpy(pout)
        else: return pout

def __make_S_by_k__(k:int):
    """Make S matrix for korder cubic interpolation - as described in [1]

    [1] S. K. Præsius and J. Arendt Jensen, “Fast Spline Interpolation using GPU Acceleration,” in 2024 IEEE Ultrasonics, Ferroelectrics, and Frequency Control Joint Symposium (UFFC-JS), Sep. 2024, pp. 1–5. doi: 10.1109/UFFC-JS60046.2024.10793976.
    """
    import numpy as np
    # make C matrix
    c_00 = np.ones(k)
    c_00[1:-1] = 4
    c_p1 = np.ones(k-1)
    c_p1[0] = 2
    c_n1 = np.flip(c_p1)

    C  = np.diag(c_00, k= 0)
    C += np.diag(c_p1, k= 1)
    C += np.diag(c_n1, k=-1)

    # make P matrix
    p_p1 = 3*np.ones(k-1)
    p_p1[0] = 2
    p_n1 = -np.flip(p_p1)

    P  = np.diag(p_p1, k= 1)
    P += np.diag(p_n1, k=-1)
    P[0,0] = -2.5
    P[-1,-1] = 2.5
    P[0,2] = 0.5
    P[-1,-3] = -0.5

    return np.linalg.inv(C) @ P