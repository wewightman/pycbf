"""Base beamforming classes implemented for the GPU"""
from dataclasses import dataclass, field
from pycbf.__bf_base_classes__ import Synthetic, Tabbed, Beamformer, BeamformerException, __BMFRM_PARAMS__, __PYCBF_DATATYPE__

from numpy import ndarray as npNDArray
from cupy  import ndarray as cpNDArray

@dataclass(kw_only=True)
class __GPU_Beamformer__(Beamformer):
    """Base class for all GPU beamformers
    """
    t0 : float = field(init=True)
    dt : float = field(init=True)
    nt :   int = field(init=True)

    nthread : int = 512
    interp : dict = field(default_factory=lambda:{"kind":"korder_cubic", "k":8})

    def __post_init__(self):
        Beamformer.__post_init__(self)

    def __check_indexing_limits__(self):
        ntxrxp = self.ntx*self.nrx*self.nop
        ntxrxp_max = 2**63 - 1
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

    def __check_or_init_buffer__(self, buffer : cpNDArray | None = None) -> cpNDArray:
        """Validate input buffer and make sure it is the right size for the given sumtype"""
        import cupy as cp
        import numpy as np

        # determine output shape based on summing choice
        if   self.sumtype ==      'none': shape = (self.ntx, self.nrx, self.nop)
        elif self.sumtype ==   'tx_only': shape = (          self.nrx, self.nop)
        elif self.sumtype ==   'rx_only': shape = (self.ntx,           self.nop)
        elif self.sumtype == 'tx_and_rx': shape =                      self.nop
        else: raise BeamformerException("Type must be 'none', 'tx_only', 'rx_only', or 'tx_and_rx'")

        if buffer is None: 
            pout = cp.zeros(shape, dtype=__PYCBF_DATATYPE__)
        else: raise Exception("Something is wrong with input buffers")

        return pout
    
    def __run_interp_type__(self, txrxt, pout): raise NotImplementedError("You must implement '__run_interp_type__' for class GPUBeamformer")
    
    def __call__(self, txrxt : cpNDArray | npNDArray, out_as_numpy : bool = True, buffer : cpNDArray | None = None) -> cpNDArray | npNDArray:

        import cupy as cp
        import numpy as np

        # format txrxt based on type and shape
        if isinstance(txrxt, npNDArray):
            txrxt = cp.ascontiguousarray(cp.array(txrxt), dtype=__PYCBF_DATATYPE__)
            
        elif isinstance(txrxt, cpNDArray):
            if (txrxt.dtype != __PYCBF_DATATYPE__) or (txrxt.dtype != cp.float32) or (txrxt.dtype != cp.float64):
                raise BeamformerException("Cupy array dtype must be either cupy or numpy float 32")
            
            if not txrxt.flags['C_CONTIGUOUS']:
                txrxt = cp.ascontiguousarray(txrxt, dtype=__PYCBF_DATATYPE__)

        else:
            raise BeamformerException("txrxt must be an instance of either a cupy or numpy ndarray but was ", type(txrxt))
        
        if txrxt.shape != (self.ntx, self.nrx, self.nt):
            shapestr = ""
            for dim in txrxt.shape: shapestr += f"{dim:d}, "
            shapestr = shapestr[:-2]

            raise BeamformerException(
                f"'txrxt' must be ({self.ntx}, {self.nrx}, {self.nt}) based on input parameters but was ({shapestr})"
            )

        # validate that input buffor is correct format or make new one
        pout = __GPU_Beamformer__.__check_or_init_buffer__(self, buffer)
        
        # beamform the data with specified summing
        self.__run_interp_type__(txrxt, pout)

        # return as numpy or cupy array depending on call specification
        if out_as_numpy: return cp.asnumpy(pout)
        else: return pout

@dataclass(kw_only=True)
class SyntheticBeamformer(Synthetic, __GPU_Beamformer__):
    """GPU implementation of synthetic-point based beamformers
    """
    def __post_init__(self):
        __GPU_Beamformer__.__post_init__(self)
        Synthetic.__post_init__(self)
        self.__check_indexing_limits__()

        from cupy import array, ascontiguousarray
        from pycbf.gpu.__engines__ import RFInfo
        import numpy as np

        params = __BMFRM_PARAMS__[self.id]

        # copy TX parameters into shared GPU memory
        params['ovectx'] = ascontiguousarray(array(self.ovectx), dtype=__PYCBF_DATATYPE__)
        params['nvectx'] = ascontiguousarray(array(self.nvectx), dtype=__PYCBF_DATATYPE__)
        params[ 'doftx'] = ascontiguousarray(array(self. doftx), dtype=__PYCBF_DATATYPE__)
        params[ 'alatx'] = ascontiguousarray(array(self. alatx), dtype=__PYCBF_DATATYPE__)
        params[  't0tx'] = ascontiguousarray(array(self.  t0tx), dtype=__PYCBF_DATATYPE__)

        # copy RX parameters into shared GPU memory
        params['ovecrx'] = ascontiguousarray(array(self.ovecrx), dtype=__PYCBF_DATATYPE__)
        params['nvecrx'] = ascontiguousarray(array(self.nvecrx), dtype=__PYCBF_DATATYPE__)
        params[ 'alarx'] = ascontiguousarray(array(self. alarx), dtype=__PYCBF_DATATYPE__)

        # copy output pnts into shared GPU memory
        params[  'pnts'] = ascontiguousarray(array(self.  pnts), dtype=__PYCBF_DATATYPE__)

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

    def __run_interp_type__(self, txrxt : cpNDArray, pout : cpNDArray):
        """Based on the interpolation type, use the correct engine
        """
        import cupy as cp
        import numpy as np

        sumtypes = {'none':0, 'tx_only':1, 'rx_only':2, 'tx_and_rx':3}

        if self.interp['kind'] == 'korder_cubic':
            from pycbf.gpu.__engines__ import das_bmode_synthetic_korder_cubic as gpu_kernel

            k = int(self.interp['k'])
            S = cp.ascontiguousarray(cp.array(__make_S_by_k__(k)), dtype=__PYCBF_DATATYPE__)

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
                np.int32(k), S,
                __PYCBF_DATATYPE__(self.c0),
                np.int64(self.nop),
                bf_params[  'pnts'],
                pout,
                np.int32(sumtypes[self.sumtype])
            )

            nblock = int(np.ceil(self.ntx * self.nrx * self.nop / self.nthread))

            gpu_kernel((nblock,1,1), (self.nthread,1,1), routine_params)

        elif self.interp['kind'] == "cubic":
            raise NotImplementedError("Ideal cubic interpolation has not been implemented")

        else:
            from pycbf.gpu.__engines__ import das_bmode_synthetic_multi_interp as gpu_kernel

            interp_keys = {"nearest":0, "linear":1, "akima":2, "makima":3}

            usf = int(self.interp['usf'])
            if usf > 1: raise NotImplementedError("upsampling has not been implemented")
                
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
                np.int32(interp_keys[self.interp['kind']]),
                __PYCBF_DATATYPE__(self.c0),
                np.int64(self.nop),
                bf_params[  'pnts'],
                pout,
                np.int32(sumtypes[self.sumtype])
            )

            nblock = int(np.ceil(self.ntx * self.nrx * self.nop / self.nthread))

            gpu_kernel((nblock,1,1), (self.nthread,1,1), routine_params)

    def __del__(self):
        if hasattr(self, "id"):
            params = __BMFRM_PARAMS__[self.id]

            # delete globally stored data if it exists
            for key in ['ovectx', 'nvectx', 'doftx', 'alatx', 't0tx', 'ovecrx', 'nvecrx', 'alarx', 'pnts']:
                if key in params.keys(): del params[key]

            del __BMFRM_PARAMS__[self.id]

@dataclass(kw_only=True)
class TabbedBeamformer(Tabbed,__GPU_Beamformer__):
    """GPU-based implementation of tabbed beamformers reading from procomputed delay tabs and apodizations
    """
    def __post_init__(self):
        __GPU_Beamformer__.__post_init__(self)
        Tabbed.__post_init__(self)
        self.__check_indexing_limits__()

        from cupy import array, ascontiguousarray
        from pycbf.gpu.__engines__ import RFInfo
        import numpy as np

        # Access the global shared buffer
        params = __BMFRM_PARAMS__[self.id]

        # copy tx/rx/output point dimensions
        params['nop']    = self.nop

        # make struct describing data
        params['rfinfo'] = np.zeros(1, dtype=RFInfo)
        params['rfinfo']['ntx']  = self.ntx
        params['rfinfo']['nrx']  = self.nrx
        params['rfinfo']['ndim'] = 0
        params['rfinfo']['tInfo']['x0'] = self.t0
        params['rfinfo']['tInfo']['dx'] = self.dt
        params['rfinfo']['tInfo']['nx'] = self.nt

        # Copy beamforming tabs to CPU memory
        params['tautx' ] = ascontiguousarray(array(self.tautx ), dtype=__PYCBF_DATATYPE__)
        params['taurx' ] = ascontiguousarray(array(self.taurx ), dtype=__PYCBF_DATATYPE__)
        params['apodtx'] = ascontiguousarray(array(self.apodtx), dtype=__PYCBF_DATATYPE__)
        params['apodrx'] = ascontiguousarray(array(self.apodrx), dtype=__PYCBF_DATATYPE__)

        __BMFRM_PARAMS__[self.id] = params
    
    def __run_interp_type__(self, txrxt : cpNDArray, pout : cpNDArray):
        """Based on the interpolation type, use the correct engine"""
        import cupy as cp
        import numpy as np

        sumtypes = {'none':0, 'tx_only':1, 'rx_only':2, 'tx_and_rx':3}

        if self.interp['kind'] == 'korder_cubic':
            from pycbf.gpu.__engines__ import das_bmode_tabbed_korder_cubic as gpu_kernel

            k = int(self.interp['k'])
            S = cp.ascontiguousarray(cp.array(__make_S_by_k__(k)), dtype=__PYCBF_DATATYPE__)

            bf_params = __BMFRM_PARAMS__[self.id]
            routine_params = (
                bf_params['rfinfo'],
                txrxt,
                bf_params['tautx'],
                bf_params['apodtx'],
                bf_params['taurx'],
                bf_params['apodrx'],
                np.int32(k), S,
                np.int64(self.nop),
                pout,
                np.int32(sumtypes[self.sumtype])
            )

            nblock = int(np.ceil(self.ntx * self.nrx * self.nop / self.nthread))

            gpu_kernel((nblock,1,1), (self.nthread,1,1), routine_params)

        elif self.interp['kind'] == "cubic":
            raise NotImplementedError("Ideal cubic interpolation has not been implemented")

        else:
            from pycbf.gpu.__engines__ import das_bmode_tabbed_multi_interp as gpu_kernel

            interp_keys = {"nearest":0, "linear":1, "akima":2, "makima":3}

            usf = int(self.interp['usf'])
            if usf > 1: raise NotImplementedError("upsampling has not been implemented")
                
            bf_params = __BMFRM_PARAMS__[self.id]
            routine_params = (
                bf_params['rfinfo'],
                txrxt,
                bf_params['tautx'],
                bf_params['apodtx'],
                bf_params['taurx'],
                bf_params['apodrx'],
                np.int32(interp_keys[self.interp['kind']]),
                np.int64(self.nop),
                pout,
                np.int32(sumtypes[self.sumtype])
            )

            nblock = int(np.ceil(self.ntx * self.nrx * self.nop / self.nthread))

            gpu_kernel((nblock,1,1), (self.nthread,1,1), routine_params)

    def __del__(self):
        if hasattr(self, "id"):
            params = __BMFRM_PARAMS__[self.id]

            # delete globally stored data if it exists
            for key in ['tautx', 'taurx', 'apodtx', 'apodrx']:
                if key in params.keys(): del params[key]

            del __BMFRM_PARAMS__[self.id]

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