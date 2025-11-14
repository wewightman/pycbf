"""All base classes for CPU-based beamformers"""
from dataclasses import dataclass, field
from pycbf.__bf_base_classes__ import Tabbed, Beamformer, BeamformerException, __BMFRM_PARAMS__
from numpy import ndarray

@dataclass(kw_only=True)
class CPUBeamformer(Beamformer):
    """Base class for CPU beamformers"""
    t0 : float = field(init=True)
    dt : float = field(init=True)
    nt :   int = field(init=True)

    nwrkr : int = 1
    thresh : float = 1E-2
    interp : dict = field(default_factory=lambda:{"kind":"nearest", "usf":8})

    def __post_init__(self):
        Beamformer.__post_init__(self)

    def __check_indexing_limits__(self):
        ntxrxp = self.ntx*self.nrx*self.nop
        ntxrxp_max = 2**31 - 1
        if ntxrxp > ntxrxp_max:
            raise BeamformerException(
                f"Too many pixels {self.nop} were requested for {self.ntx} tx and {self.nrx} rx events on a int32 based GPU. "
                + f"Product of ntx * nrx * npoints must be less than {ntxrxp_max} but was {ntxrxp}."
                )

        ntxrxt = self.ntx*self.nrx*self.nt
        ntxrxt_max = 2**31 - 1

        if ntxrxt > ntxrxt_max:
            raise BeamformerException(
                f"Too many time points {self.nt} were given for {self.ntx} tx and {self.nrx} rx events on a int32 based GPU. "
                + f"Product of ntx * nrx * nt must be less than {ntxrxt_max} but was {ntxrxt}."
                )
        
    def __get_buffer_size__(self) -> int:
        from numpy import prod
        # determine output shape based on summing choice
        if   self.sumtype ==      'none': shape = (self.ntx, self.nrx, self.nop)
        elif self.sumtype ==   'tx_only': shape = (          self.nrx, self.nop)
        elif self.sumtype ==   'rx_only': shape = (self.ntx,           self.nop)
        elif self.sumtype == 'tx_and_rx': shape =                      self.nop
        else: raise BeamformerException("Type must be 'none', 'tx_only', 'rx_only', or 'tx_and_rx'")

        return prod(shape, dtype=int)
    
    @staticmethod
    def __get_buffer_offset__(id, itx, irx) -> int:
        params  = __BMFRM_PARAMS__[id]
        sumtype = params['sumtype']
        nrx     = params['nrx']
        nop     = params['nop']

        if   sumtype ==    'none': offset = nop * (irx + itx*nrx)
        elif sumtype == 'tx_only': offset = nop * irx
        elif sumtype == 'rx_only': offset = nop * itx
        else:                      offset = 0

        return offset

@dataclass(kw_only=True)
class TabbedBeamformer(Tabbed, CPUBeamformer):
    """Base class for CPU-implemented `Tabbed` beamformers"""
    def __post_init__(self):
        CPUBeamformer.__post_init__(self)
        Tabbed.__post_init__(self)
        self.__check_indexing_limits__()

        from multiprocessing import RawArray
        from ctypes import c_float, POINTER
        from numpy import ascontiguousarray, zeros

        # Access the global shared buffer
        params = dict()

        # copy tx/rx/output point dimensions
        params['ntx']     = self.ntx
        params['nrx']     = self.nrx
        params['nop']     = self.nop
        params['t0']      = self.t0
        params['dt']      = self.dt
        params['nt']      = self.nt
        params['thresh']  = self.thresh
        params['sumtype'] = self.sumtype

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
                params['results'][ii] = RawArray(c_type, self.__get_buffer_size__())

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
            params['results'][0] = zeros(self.__get_buffer_size__(), dtype=c_type).ctypes.data_as(POINTER(c_type))

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
        from pycbf.cpu.__engine__ import __cpu_pycbf__
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

        txoff  = ravel_multi_index((itx,0), (ntx,nop))
        rxoff  = ravel_multi_index((irx,0), (nrx,nop))
        rfoff  = ravel_multi_index((itx,irx,0), (ntx,nrx,nt))
        opoff = TabbedBeamformer.__get_buffer_offset__(id, itx, irx)

        pttx   = TabbedBeamformer.__offset_pnt__(params['pttx'],           txoff)
        ptrx   = TabbedBeamformer.__offset_pnt__(params['ptrx'],           rxoff)
        patx   = TabbedBeamformer.__offset_pnt__(params['patx'],           txoff)
        parx   = TabbedBeamformer.__offset_pnt__(params['parx'],           rxoff)
        psig   = TabbedBeamformer.__offset_pnt__(params['psig'],           rfoff)
        out    = TabbedBeamformer.__offset_pnt__(params['results'][iwrkr], opoff)

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
            self.pool.starmap(TabbedBeamformer.__beamform_single__, iterator)
        else:
            for id, itx, irx in iterator:
                TabbedBeamformer.__beamform_single__(id, itx, irx)


        temp = array([params['results'][id][:self.__get_buffer_size__()] for id in range(self.nwrkr)])

        return sum(temp, axis=0)

    def __del__(self):
        if hasattr(self, "id"): del __BMFRM_PARAMS__[self.id]