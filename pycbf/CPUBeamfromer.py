from dataclasses import dataclass, field
from pycbf.__bf_base_classes__ import Tabbed, Parallelized, BeamformerException, __BMFRM_PARAMS__
from numpy import ndarray

@dataclass(kw_only=True)
class CPUBeamformer(Tabbed, Parallelized):
    nwrkr : int = 1
    t0 : float = field(init=True)
    dt : float = field(init=True)
    nt :   int = field(init=True)
    thresh : float = 1E-2
    interp : dict = field(default_factory=lambda:{"kind":"cubic"})

    def __post_init__(self):
        Parallelized.__post_init__(self)
        Tabbed.__post_init__(self)
        from multiprocessing import RawArray
        from ctypes import c_float, POINTER
        from numpy import ascontiguousarray, zeros

        # Access the global shared buffer
        params = dict()

        # copy tx/rx/output point dimensions
        params['ntx']    = self.ntx
        params['nrx']    = self.nrx
        params['nop']    = self.nop
        params['t0']     = self.t0
        params['dt']     = self.dt
        params['nt']     = self.nt
        params['thresh'] = self.thresh

        kind = self.interp.get("kind", '')
        if kind == 'cubic':
            params['interp'] = self.interp
        elif kind == 'nearest':
            usf = int(self.interp.get("usf", 1))
            if usf < 1: raise BeamformerException("'usf' must be an integer >= 1 for 'interp' type 'nearest'")
            self.interp['usf'] = usf
            params['interp'] = self.interp
        else:
            raise BeamformerException("'interp[\"kind\"]' must be either 'cubic' or 'nearest'")

        # the ctype being used, might eb flexible in future
        c_type = c_float

        if self.nwrkr > 1:
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

            self.__start_pool__()
        else:
            # copy the tabs
            params['pttx'] = ascontiguousarray(self. tautx, dtype=c_type).ctypes.data_as(POINTER(c_type))
            params['ptrx'] = ascontiguousarray(self. taurx, dtype=c_type).ctypes.data_as(POINTER(c_type))
            params['patx'] = ascontiguousarray(self.apodtx, dtype=c_type).ctypes.data_as(POINTER(c_type))
            params['parx'] = ascontiguousarray(self.apodrx, dtype=c_type).ctypes.data_as(POINTER(c_type))

            # build an output buffer for each worker
            params['results'] = {}
            params['results'][0] = zeros(self.nop, dtype=c_type).ctypes.data_as(POINTER(c_type))

            # build the buffer for input RF data
            params['psig'] = zeros(self.nt * self.ntx * self.nrx, dtype=c_type).ctypes.data_as(POINTER(c_type))

            params['idx'] = 0

        __BMFRM_PARAMS__[self.id] = params

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

        interp = params['interp']

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

        if interp['kind'] == 'cubic':
            __cpu_pycbf__.beamform_cubic(
                c_float(t0), c_float(dt), c_int(nt), psig,
                c_int(nop), c_float(thr), pttx, patx, ptrx, parx, out
            )
        elif interp['kind'] == 'nearest':
            usf = interp['usf']
            __cpu_pycbf__.beamform_nearest(
                c_float(t0), c_float(dt), c_int(nt), psig,
                c_int(nop), c_float(thr), pttx, patx, ptrx, parx, out, c_int(usf)
            )

    @staticmethod
    def __mp_init_workers__(id, queue):
        __BMFRM_PARAMS__[id]['idx'] = queue.get()

    def __start_pool__(self):
        """Make the multiprocessing pool"""
        from multiprocessing import Pool, Manager

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

    def __zero_buffers__(self):
        from ctypes import memset, sizeof
        params = __BMFRM_PARAMS__[self.id]
        for iwrkr in range(self.nwrkr):
            memset(params['results'][iwrkr], 0, int(self.nop*sizeof(params['results'][iwrkr]._type_)))
        
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

        self.__zero_buffers__()

        # delay and apodize
        iterator = product([self.id], range(self.ntx), range(self.nrx))
        if self.nwrkr > 1:
            self.pool.starmap(CPUBeamformer.__beamform_single__, iterator)
        else:
            for id, itx, irx in iterator:
                CPUBeamformer.__beamform_single__(id, itx, irx)


        temp = array([params['results'][id][:self.nop] for id in range(self.nwrkr)])

        return sum(temp, axis=0)

    def __del__(self):
        del __BMFRM_PARAMS__[self.id]