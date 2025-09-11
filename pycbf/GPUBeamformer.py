from dataclasses import dataclass, field
from pycbf.__bf_base_classes__ import SynthPointed, Parallelized, BeamformerException, __BMFRM_PARAMS__

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

        from cupy import ascontiguousarray, float32
        from pycbf.gpu import RFInfo
        import numpy as np
        params = dict()

        # copy TX parameters into shared memory
        params['ovectx'] = ascontiguousarray(self.ovectx, dtype=float32)
        params['nvectx'] = ascontiguousarray(self.nvectx, dtype=float32)
        params[ 'doftx'] = ascontiguousarray(self. doftx, dtype=float32)
        params[ 'alatx'] = ascontiguousarray(self. alatx, dtype=float32)
        params[  't0tx'] = ascontiguousarray(self.  t0tx, dtype=float32)

        # copy RX parameters into shared memory
        params['ovecrx'] = ascontiguousarray(self.ovecrx, dtype=float32)
        params['nvecrx'] = ascontiguousarray(self.nvecrx, dtype=float32)
        params[ 'alarx'] = ascontiguousarray(self. alarx, dtype=float32)

        # copy output pnts into shared memory
        params[  'pnts'] = ascontiguousarray(self.  pnts, dtype=float32)

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

    def __call__(self, txrxt, pout = None):
        pass