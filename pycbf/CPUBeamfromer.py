from dataclasses import dataclass, field
from pycbf.__bf_base_classes__ import Tabbed, Parallelized, BeamformerException, __BMFRM_PARAMS__
from numpy import ndarray

@dataclass(kw_only=True)
class CPUBeamformer(Tabbed, Parallelized):
    nwrkr : int = 8
    t0 : float = field(init=True)
    dt : float = field(init=True)
    nt :   int = field(init=True)
    thresh : float = 1E-2

    def __post_init__(self):
        Parallelized.__post_init__(self)
        Tabbed.__post_init__(self)
        from multiprocessing import RawArray
        from ctypes import c_float

        __BMFRM_PARAMS__[self.id] = {}

        # Access the global shared buffer
        params = __BMFRM_PARAMS__[self.id]

        # copy tx/rx/output point dimensions
        params['ntx']    = self.ntx
        params['nrx']    = self.nrx
        params['nop']    = self.nop
        params['t0']     = self.t0
        params['dt']     = self.dt
        params['nt']     = self.nt
        params['thresh'] = self.thresh

        # the ctype being used, might eb flexible in future
        c_type = c_float

        # copy the tabs
        params['pttx'] = RawArray(c_type, self. tautx.flatten())
        params['ptrx'] = RawArray(c_type, self. taurx.flatten())
        params['patx'] = RawArray(c_type, self.apodtx.flatten())
        params['parx'] = RawArray(c_type, self.apodrx.flatten())

        # build an output buffer for each worker
        params['results'] = {}
        for ii in range(self.nwrkr):
            params['results'][ii] = RawArray(c_type, self.nop)

        # build the buffer for input RF data
        params['psig'] = RawArray(c_type, self.nt * self.ntx * self.nrx)

        self.pool = None

    @staticmethod
    def __offset_pnt__(pnt, offset:int):
        from ctypes import sizeof, cast, POINTER, c_void_p

        # convert to a void-type pointer object
        pvoid = cast(pnt, c_void_p)

        # calculate the element offset in bytes
        pvoid.value += int(offset * sizeof(pnt._type_))

        # reform the pointer as its original type
        pnt_out = cast(pvoid, POINTER(pnt._type_))

        return pnt_out


    @staticmethod
    def __beamform_single__(id, itx, irx):
        """Use the identified beamformer to beamform the aline specified by tx and rx indices"""
        from numpy import ravel_multi_index
        from pycbf.cpu import __cpu_pycbf__
        from ctypes import c_float, c_int

        params = __BMFRM_PARAMS__[id]

        iwrkr = params['idx']
        ntx   = params['ntx']
        nrx   = params['nrx']
        nop   = params['nop']

        t0    = params['t0']
        dt    = params['dt']
        nt    = params['nt']
        thr   = params['thresh']

        txoff = ravel_multi_index((itx,0), (ntx,nop))
        rxoff = ravel_multi_index((irx,0), (nrx,nop))
        rfoff = ravel_multi_index((itx,irx,0), (ntx,nrx,nt))

        pttx = CPUBeamformer.__offset_pnt__(params['pttx'], txoff)
        ptrx = CPUBeamformer.__offset_pnt__(params['ptrx'], rxoff)
        patx = CPUBeamformer.__offset_pnt__(params['patx'], txoff)
        parx = CPUBeamformer.__offset_pnt__(params['parx'], rxoff)
        psig = CPUBeamformer.__offset_pnt__(params['psig'], rfoff)

        out  = params['results'][iwrkr]

        __cpu_pycbf__.beamform(
            c_float(t0), c_float(dt), c_int(nt), psig,
            c_int(nop), c_float(thr), pttx, patx, ptrx, parx, out
        )

    @staticmethod
    def __mp_init_workers__(id, queue):
        __BMFRM_PARAMS__[id]['idx'] = queue.get()

    def __start_pool__(self):
        """Make the multiprocessing pool"""
        from multiprocessing import Pool, Manager

        if (self.pool is not None): self.pool.close()

        # use a process-shared queue to link workers with a persistent buffer
        manager = Manager()
        idQueue = manager.Queue()
        for i in range(self.nwrkr): idQueue.put(i)

        # setup the pool
        self.pool = Pool(
            self.nwrkr, 
            initializer=self.__mp_init_workers__,
            initargs=(self.id, idQueue,)
        )

    def __kill_pool__(self):
        if hasattr(self, "pool"): self.pool.terminate()

    def __zero_buffers__(self):
        from ctypes import memset, sizeof
        params = __BMFRM_PARAMS__[self.id]
        for iwrkr in range(self.nwrkr):
            memset(params['results'][iwrkr], 0, int(self.nop*sizeof(params['results'][iwrkr]._type_)))
        
    def __call__(self, txrxt:ndarray):
        from numpy import array, sum
        from itertools import product

        # ensure input data meets data specs
        if txrxt.shape != (self.ntx, self.nrx, self.nt):
            raise BeamformerException(f"Input data must be {self.ntx} by {self.nrx} by {self.nt}")
        
        params = __BMFRM_PARAMS__[self.id]

        for ii, rf in enumerate(txrxt.flatten()): params['psig'][ii] = rf

        self.__zero_buffers__()

        # delay and apodize
        self.__start_pool__()
        self.pool.starmap(CPUBeamformer.__beamform_single__, product([self.id], range(self.ntx), range(self.nrx)))

        temp = array([params['results'][id][:self.nop] for id in range(self.nwrkr)])

        return sum(temp, axis=0)

    def __del__(self):
        self.__kill_pool__()