from dataclasses import dataclass, field
from pycbf.__bf_base_classes__ import SynthPointed, Parallelized, BeamformerException, __BMFRM_PARAMS__

from numpy import ndarray as npNDArray
from cupy  import ndarray as cpNDArray

@dataclass(kw_only=True)
class GPUBeamformer(SynthPointed, Parallelized):
    t0 : float = field(init=True)
    dt : float = field(init=True)
    nt :   int = field(init=True)
    c0 : float = field(init=True)
    nthread : int = 512

    def __post_init__(self):
        Parallelized.__post_init__(self)
        SynthPointed.__post_init__(self)

        from cupy import array, ascontiguousarray, float32
        from pycbf.gpu import RFInfo
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
        from pycbf.gpu import das_bmode_cubic

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
        das_bmode_cubic((nblock,1,1), (self.nthread,1,1), routine_params)

        if out_as_numpy: return cp.asnumpy(pout)
        else: return pout

    def __del__(self):
        params = __BMFRM_PARAMS__[self.id]

        # delete globally stored data if it exists
        for key in ['ovectx', 'nvectx', 'doftx', 'alatx', 't0tx', 'ovecrx', 'nvecrx', 'alarx', 'pnts']:
            if key in params.keys(): del params[key]

        del __BMFRM_PARAMS__[self.id]