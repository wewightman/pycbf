"""Basic beamforming base classes to build all specific beamforming classes from. Specific implementation details can be found in the hardware-specific modules `pycbf.cpu` and `pycbf.gpu`"""
from dataclasses import dataclass, field
from typing import ClassVar, Literal
from numpy import ndarray
import logging
logger = logging.getLogger(__name__)

global __BMFRM_PARAMS__
__BMFRM_PARAMS__ = {}

class BeamformerException(Exception): pass

@dataclass(kw_only=True)
class Beamformer():
    """The base beamforming class for all beamformers in this repository
    
    This class registers an instance of a `Beamformer` in global memory so no matter the parallelization technique or hardware being used, the parameters can be stored ina global location. This class automatically generates an `id` field on instantiation.
    It also requires the user to identify how the delayed data is summed and what kind of interpolation is being used.

    # Defining `sumtype`
    The `sumtype` keyword parameter determines which axes are being summed over inside of the optimized C or CUDA extension libraries.

    - To make a simple delay and sum (DAS) beamformer, you must sum over all tx and rx events - which can be achieved in kernel using `sumtype='tx_and_rx'`, allowing the use of smaller output buffers.
    - To keep all tx and rx events separate, use `sumtype='none'`. This output can be used for many advanced beamforming techniques - but requires very large output buffers.
    - To balance the optimization of output buffer size and advanced beamforming flexibility, two other options - `sumtype='rx_only'` and `sumtype='tx_only'` can be used

    # Defining the interpolation engine
    All beamfomrers in this repository should have the same available interpolation options (eventually). The options and sub options are as follows...

    The interpolation method can be user defined with the parameter `interp` - a dictionary with a `kind` key with one of the following values...
    - `nearest`: (CPU and GPU) Extracts the nearest sample of the pressure signal to the delay tab. 
        - (CPU only) If optional integer upsample factor parameter `usf` > 1, upsamples the RF channel data using ideal `cubic` interpolation 
    - `linear`: (GPU only) does linear interpolation between adjacent points
    - `korder_cubic`: (GPU only) 1D cubic spline method described by Præsius and Jensen.
        - Based on integer parameter `k`, estimates signal first derivatives using a convolution kernel of length `k` where `k` > 4 and even
        - Numerical interpolation noise reduces with increasing `k` - but leads to slower computation times
    - `akima`: (GPU only) Akima cubic interpolation uses five signal points at most
    - `makima`: (GPU only) modified Akima (mAkima) cubic interpolation uses five signal points at most but is smoother than akima
    - `cubic`: (CPU only) Refers to exact cubic hermite spline interpolation with signal second derivatives estimated from the entire signal trace
    """
    nbf : ClassVar[int] = 0
    id  : int = field(init=False)
    interp : dict = field(default_factory=lambda:{"kind":"cubic"})
    sumtype : Literal['none', 'tx_only', 'rx_only', 'tx_and_rx'] = 'tx_and_rx'

    ntx: int = field(init=False)
    nrx: int = field(init=False)
    nop: int = field(init=False)

    def __post_init__(self):
        self.id = Beamformer.nbf
        Beamformer.nbf += 1

        __BMFRM_PARAMS__[self.id] = {}

        self.__check_interp__()
        self.__check_sumtype__()
    
    def __check_sumtype__(self):
        valid_keys = ['none', 'tx_only', 'rx_only', 'tx_and_rx']

        err_msg = f"sumtype was {self.sumtype} but must be one of the following: "

        for key in valid_keys: err_msg += key + ", "
        err_msg = err_msg[:-2]
        
        if self.sumtype not in valid_keys: raise BeamformerException(err_msg)

    def __check_interp__(self):
        valid_kinds = ["nearest", "linear", "akima", "makima", "korder_cubic", "cubic"]
        kind = self.interp.get('kind', None)
        if (kind is None) or (kind not in valid_kinds): 
            err_msg = f"interp['kind'] was {kind} but must be one of the following: "

            for key in valid_kinds: err_msg += key + ", " 
            err_msg = err_msg[:-2]

            raise BeamformerException(err_msg)
        
        elif kind == 'korder_cubic':
            k = self.interp.get("k", None)
            if k is None: self.interp['k'] = 4
            if (k < 4) or (k%2 != 0): raise BeamformerException("'k' for interp kind 'korder_cubic' must be at least four and even")

        elif kind == 'cubic': pass

        else:
            usf = self.interp.get("usf", None)
            if usf is None: self.interp['usf'] = 1
            elif (usf < 1): raise BeamformerException(f"'usf' for interp kind '{self.interp['kind']}' must be an interger greater than 1")

@dataclass(kw_only=True)
class Tabbed():    
    """The minimum inputs for a `Tabbed` beamformer are four matrices: `tautx`, `taurx`, `apodtx`, and `apodrx`. 
    If `Np` is the number of beamforming points, `Ntx` is the number of transmit events, and `Nrx` is the number of receive events...
    - `tautx` and `apodtx` both have the shape `Ntx` by `Np`
    - `taurx` and `apodrx` both have the shape `Nrx` by `Np`
    """
    tautx  : ndarray = field(init=True)
    taurx  : ndarray = field(init=True)
    apodtx : ndarray = field(init=True)
    apodrx : ndarray = field(init=True)
    ntx : int = field(init=False)
    nrx : int = field(init=False)
    nop : int = field(init=False)

    def __post_init__(self):
        from numpy import ndim

        # check that all inputs are 2D
        all2d = (ndim(self.tautx) == 2) and (ndim(self.taurx) == 2) and (ndim(self.apodtx) == 2) and (ndim(self.apodrx) == 2)

        # extract dimensions of tx/rx/output point array
        ntx = self.tautx.shape[0]
        nrx = self.taurx.shape[0]
        nop = self.tautx.shape[1]

        # ensure dimensions are consistent
        alltxdim = (self.apodtx.shape[0] == ntx)
        allrxdim = (self.apodrx.shape[0] == nrx)
        allopdim = (self.taurx.shape[1] == nop) and (self.apodtx.shape[1] == nop) and (self.apodtx.shape[1] == nop)

        if not all2d:
            raise BeamformerException(f"all delay tabs and apodizations must be 2D matrices")
        if not alltxdim:
            raise BeamformerException(f"tautx and apodtx must have the same shape in dim 0 but were {self.tautx.shape[0]} and {self.apodtx.shape[0]}")
        if not allrxdim:
            raise BeamformerException(f"taurx and apodrx must have the same shape in dim 0 but were {self.taurx.shape[0]} and {self.apodrx.shape[0]}")
        if not allopdim:
            raise BeamformerException(f"tautx, taurx, apodtx, and apodrx must have the same shape in dim 1 but were {self.tautx.shape[0]}, {self.taurx.shape[0]}, {self.apodtx.shape[0]}, and {self.apodrx.shape[0]}")
        
        self.ntx = ntx
        self.nrx = nrx
        self.nop = nop

@dataclass(kw_only=True)
class Synthetic():   
    """The `Synthetic` specific parameters are...
    - `ovectx`: the origin of the transmit point
        - an Ntx by 2 or 3 matrix
    - `nvectx`: the normal vector of the transmit point indicating the direction of wave propagation
        - Ntx by 2 or 3 matrix
    - `doftx`: the depth of field around `ovectx` to beamform parallel to `nvectx`
        - allows for more accurate reconstruction of the physically realistic DOF around a focal point with focused transmits
        - length Ntx vector
    - `alatx`: the acceptance angle/directivity of the transmit point relative to `nvectx`
        - can be used to encode things like F-number for a focused beam or to limit the beamformed FOV based on the projected source through the physical aperture
        - length Ntx vector
    - `t0tx`: the time point that the wave passes through `ovectx` relative to the input time trace
        - length Ntx vector
    - `ovecrx`: the location of the receive point
        - Nrx by 2 or 3 matrix
    - `nvecrx`: the orientation of the receive point
        - Nrx by 2 or 3 matrix
    - `alarx`: the acceptance angle/directivity of the receive point
        - length Nrx vector
    - `c0`: the assumed speed of sound in the medium.
    - `pnts`: a vector containing the points to beamform
        - Np by 2 or 3 matrix
    """
    ovectx  : ndarray = field(init=True)    # source point location (ntx by ndim matrix)
    nvectx  : ndarray = field(init=True)    # normal vector of the source point (ntx by ndim matrix)
    doftx   : ndarray = field(init=True)    # depth of field over which to use planar delay tabs around ovec (ntx length vector)
    alatx   : ndarray = field(init=True)    # angular acceptance around nvec (fnumber basically) (ntx length vector)
    t0tx    : ndarray = field(init=True)    # time the wave front passes through ovectx (ntx length vector)

    ovecrx  : ndarray = field(init=True)    # location of the recieve sensors (nrx by ndim matrix)
    nvecrx  : ndarray = field(init=True)    # orientation of the recieve sensor (nrx by ndim matrix)
    alarx   : ndarray = field(init=True)    # acceptance angle of the recieve sensor (nrx length vector)
    
    c0 : float = field(init=True)           # media SOS 

    pnts    : ndarray = field(init=True)    # number of reconstruction points (nop by ndim matrix)

    ntx     : int = field(init=False)       # number of transmit events
    nrx     : int = field(init=False)       # number of recieve events
    nop     : int = field(init=False)       # number of points to recon
    ndimp   : int = field(init=False)       # number of spatial dimensions to reconstruct over (2 or 3)

    def __post_init__(self):
        from numpy import ndim

        # check that all vector inputs are 1D
        all1d = (ndim(self.doftx) == 1) and (ndim(self.alatx) == 1) and (ndim(self.t0tx) == 1) and (ndim(self.alarx) == 1) 

        # check that all matrix inputs are 2D
        all2d = (ndim(self.ovectx) == 2) and (ndim(self.nvectx) == 2) and (ndim(self.ovecrx) == 2) and (ndim(self.nvecrx) == 2)

        # extract dimensions of tx/rx/output point array
        ntx     = self.ovectx.shape[0]
        nrx     = self.ovecrx.shape[0]
        nop     = self.pnts.shape[0]
        ndimp   = self.pnts.shape[1]

        # ensure dimensions are consistent
        alltxdim = (self.nvectx.shape[0] == ntx) and (self.doftx.shape[0] == ntx) and (self.alatx.shape[0] == ntx) and (self.t0tx.shape[0] == ntx)
        allrxdim = (self.nvecrx.shape[0] == nrx) and (self.alarx.shape[0] == nrx)
        allndimp = (self.ovectx.shape[1] == ndimp) and (self.nvectx.shape[1] == ndimp) and (self.ovecrx.shape[1] == ndimp) and (self.nvecrx.shape[1] == ndimp)

        if not all1d:
            raise BeamformerException(f"all acceptance angle, depth of field, and t0 vectors must be 1D")
        if not all2d:
            raise BeamformerException(f"all origin and normal vector matrices must be 2D")
        if not alltxdim:
            raise BeamformerException(f"All tx variables must have the same shape in the first dimension")
        if not allrxdim:
            raise BeamformerException(f"All rx variables must have the same shame in the first variable")
        if not allndimp:
            raise BeamformerException(f"All matrix variables must have the same number of dimensions")
        
        self.ntx    = ntx
        self.nrx    = nrx
        self.nop    = nop
        self.ndimp  = ndimp
