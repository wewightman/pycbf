from dataclasses import dataclass
from pycbf.__bf_base_classes__ import BeamformerException, __BMFRM_PARAMS__
from pycbf.CPUBeamfromer import CPUBeamformer
from numpy import ndarray

@dataclass(kw_only=True)
class CPUCoherenceBeamformer(CPUBeamformer):
        
    def __call__(self, txrxt:ndarray):
        from numpy import array, sum, ascontiguousarray
        from itertools import product
        from ctypes import memmove, c_float, POINTER, sizeof

        # ensure input data meets data specs
        if txrxt.shape != (self.ntx, self.nrx, self.nt):
            raise BeamformerException(f"Input data must be {self.ntx} by {self.nrx} by {self.nt}")
        
        params = __BMFRM_PARAMS__[self.id]

        rf = ascontiguousarray(txrxt, dtype=c_float).ctypes.data_as(POINTER(c_float))

        #for ii, rf in enumerate(txrxt.flatten()): params['psig'][ii] = rf
        memmove(params['psig'], rf, sizeof(c_float)*txrxt.size)

        # delay, apodize, and sum all transmissions but keep receive elements separate
        if self.nwrkr > 1:
            rxtraces = []
            for irx in range(self.nrx):
                self.__zero_buffers__()
                self.pool.starmap(CPUCoherenceBeamformer.__beamform_single__, product([self.id], range(self.ntx), [irx]))
                temp = array([params['results'][id][:self.nop] for id in range(self.nwrkr)])
                rxtraces.append(array(sum(temp, axis=0), copy=True))
        else:
            rxtraces = []
            for irx in range(self.nrx):
                self.__zero_buffers__()
                for id, itx in product([self.id], range(self.ntx)):
                    CPUCoherenceBeamformer.__beamform_single__(id, itx, irx)
                rxtraces.append(array(params['results'][0][:self.nop], copy=True))

        return array(rxtraces)