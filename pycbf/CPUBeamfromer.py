from pycbf.Beamformer import Beamformer, BeamformerException
from numpy import ndarray

class CPUBeamformer(Beamformer):
    nwrkr : int = 4

    def __post_init__(self):
        super().__post_init__()

        # Copy 

    @staticmethod
    def __beamform_single__(tautx, apodtx, taurx, apodrx, signal, t):
        """Beamform the signal from a single element for all reconstruction points"""
        from scipy.interpolate import CubicSpline

        # make the cubic interpolator
        fsig = CubicSpline(t, signal)

        # delay and apodize the data
        return apodtx * apodrx * fsig(tautx + taurx)
    
    def __map_aline__(self, txrxt, t):
        from numpy import array
        for itx in range(self.ntx):
            for irx in range(self.nrx):
                yield [
                    array(self.tautx[itx,:], copy=True),
                    array(self.apodtx[itx,:], copy=True),
                    array(self.taurx[irx,:], copy=True),
                    array(self.apodrx[irx,:], copy=True),
                    array(txrxt[itx,irx,:], copy=True),
                    array(t, copy=True)
                ]

    def __start_pool__(self):
        """Make the multiprocessing pool"""
        from multiprocessing import Pool

        if hasattr(self, "pool"): return
        self.pool = Pool()

    def __kill_pool__(self):
        if hasattr(self, "pool"): self.pool.terminate()
        
    def __call__(self, txrxt:ndarray, t:ndarray):
        # ensure input data meets data specs
        if txrxt.shape != (self.ntx, self.nrx, len(t)):
            raise BeamformerException(f"Input data must be {self.ntx} by {self.nrx} by {len(t)}")

        # delay and apodize
        self.__start_pool__()
        results = self.pool.starmap_async(CPUBeamformer.__beamform_single__, self.__map_aline__(txrxt, t))

        # sum
        im = 0
        for result in results:
            im += result

    def __del__(self):
        self.__kill_pool__()