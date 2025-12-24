"""A set of routines to calcualte the delay tabs assuming the full aperture is used in each transmit"""

def calc_rx_tabs_and_apods(cm, xele, xout, zout, fnum):
    """Calcualte the recieve delay tabs using a homogeneous speed of sound
    
    # Parameters:
    - `cm`: the speed of sound in the media in m/s
    - `xele`: the x coordinate of the center of each transmit element in m
    - `xout`: the x coordinate of the output grid in m
    - `zout`: the z coordinate of the output grid in m
    - `fnum`: the receive beamforming f-number (limits angular acceptance for a rx element)
    
    # Returns:
    - `taurx`: the receive delay tabs for each element
    - `apodrx`: the receive apodization tabs for each pixel
    """
    import numpy as np

    # coordinates of elements
    XELE, ZELE = np.meshgrid(xele, 0, indexing='ij')

    # coordinates of output grid
    XOUT, ZOUT = np.meshgrid(xout, zout, indexing='ij')

    ## Delays and apodization for receive
    # calculate the distance between each element and output point and convert from distance to time
    P = np.array([XOUT.flatten(), ZOUT.flatten()]).T
    E = np.array([XELE.flatten(), ZELE.flatten()]).T
    dX_EP = P[None,:,:]-E[:,None,:]
    taurx = np.linalg.norm(dX_EP, axis=-1)/cm

    # calculate the fnumber based apodization for each element
    apodrx = np.abs(dX_EP[...,0]) - dX_EP[...,1]/(2*fnum) < 0

    return taurx, apodrx

def calc_tx_tabs_and_apods_divergent_2D(steers, r0, ctx, cm, xele, xout, zout):
    """Calcualte the transmit delay tabs for each pixel using a homogeneous speed of sound

    This method assumes all elements are usedin transmision
    
    # Parameters:
    - `steers`: the steering angle of the synthetic source in radians
    - `r0`: the radius off the synthetic focal spot in meters (assumed to be negative)
    - `ctx`: the transmit speed of sound in m/s
    - `cm`: the speed of sound in the media in m/s
    - `xele`: the x coordinate of the center of each transmit element in m
    - `xout`: the x coordinate of the output grid in m
    - `zout`: the z coordinate of the output grid in m

    # Returns:
    - `tautx`: the transmit delay tabs for each output pixel
    - `apodtx`: the transmit apodization for each output pixel
    """
    import numpy as np

    if ctx != cm: raise Exception("Global SOS correction has not been implemented for divergent wave imaging")

    ## Generate coordintate vectors
    #   coordinates of the synthetic sources
    r0 = -np.abs(r0)
    O = np.array([r0 * np.sin(steers), r0 * np.cos(steers)]).T # Nsteer by 2

    #   coordinates of elements
    XELE, ZELE = np.meshgrid(xele, 0, indexing='ij')
    E = np.array([XELE.flatten(), ZELE.flatten()]).T # Nele by 2 

    #   coordinates of output grid
    XOUT, ZOUT = np.meshgrid(xout, zout, indexing='ij')
    P = np.array([XOUT.flatten(), ZOUT.flatten()]).T # Nout by 2

    ## Calculate the reference delay tab for each synthetic source
    dX_OE = E[None,:,:] - O[:,None,:] # Nsteer by Nele by 2
    tau0 = np.min(np.linalg.norm(dX_OE, axis=-1)/ctx, axis=1, keepdims=True) # Nsteer by 1

    ## Calcualte the delay tabs for each recon pixel
    dX_OP = P[None,:,:] - O[:,None,:] # Nout by Nsteer by 2
    tautx = np.linalg.norm(dX_OP, axis=-1)/cm - tau0

    ## Calcualte the apodization as the projection of the source through the aperture
    #   temporarily, just ones masking
    apodtx = np.ones(tautx.shape)

    return tautx, apodtx

def calc_tx_tabs_and_apods_focused_2D(steers, r0, ctx, cm, xele, xout, zout):
    """Calcualte the transmit delay tabs for each pixel using a homogeneous speed of sound

    This method assumes all elements are usedin transmision
    
    # Parameters:
    - `steers`: the steering angle of the synthetic source in radians
    - `r0`: the radius off the synthetic focal spot in meters (assumed to be negative)
    - `ctx`: the transmit speed of sound in m/s
    - `cm`: the speed of sound in the media in m/s
    - `xele`: the x coordinate of the center of each transmit element in m
    - `xout`: the x coordinate of the output grid in m
    - `zout`: the z coordinate of the output grid in m

    # Returns:
    - `tautx`: the transmit delay tabs for each output pixel
    - `apodtx`: the transmit apodization for each output pixel
    """
    import numpy as np

    from pycbf.helpers.__full_aperture_2D__ import calc_tx_synthetic_points_focused_2D

    if ctx != cm: raise Exception("Global SOS correction has not been implemented for divergent wave imaging")

    # calculate synthetic point parameters - makes it easier to calculate the delay tabs directly
    ovectx, nvectx, t0tx, _, alatx = calc_tx_synthetic_points_focused_2D(steers, ctx, cm, xele, r0)

    #   coordinates of output grid
    XOUT, ZOUT = np.meshgrid(xout, zout, indexing='ij')
    P = np.array([XOUT.flatten(), ZOUT.flatten()]).T # Nout by 2

    dX = P[None,:,:] - ovectx[:,None,:]
    dX_proj = np.sum(dX * nvectx[:,None,:], axis=-1)
    dX_mag  = np.linalg.norm(dX, axis=-1)
    tautx = np.sign(dX_proj) * dX_mag/cm + t0tx[:,None]

    print(alatx, nvectx.shape)
    dX_mag[dX_mag < 1E-12] = 1E-12
    apodtx = np.arccos(np.abs(dX_proj)/dX_mag) < alatx[:,None]
    return tautx, apodtx

def calc_tx_tabs_and_apods_pw_2D(steers, ctx, cm, xele, xout, zout):
    """Make delay tabs and apodizations for a 2D linear array
    
    # Parameters
    - `steers`: steering angles in radians
    - `ctx`: assumed speed of spound on transmission in m/s
    - `cm`: speed of sound in the medium
    - `xele`: lateral position of the elements in meters
    - `xout`: lateral position of the output grid in meters
    - `zout`: axial position of the output grid in meters

    # Returns
    - `tautx`: the transmit delay tabs
    - `apodtx`: the transmit apodization
    """

    import numpy as np

    #   coordinates of output grid
    XOUT, ZOUT = np.meshgrid(xout, zout, indexing='ij')
    P = np.array([XOUT.flatten(), ZOUT.flatten()]).T # Nout by 2

    ## delays and apodization for each plane wave transmit
    # plane wave transmit delays
    O = np.array([[xele[-1] if steer <= 0 else xele[0] for steer in steers], np.zeros(steers.shape)]).T # origin coordinates for each plane wave
    steers = np.arcsin(cm * np.sin(steers) / ctx)
    N = np.array([np.sin(steers), np.cos(steers)]).T
    dX_OP = P[None,:,:] - O[:,None,:]
    tautx = np.sum(dX_OP * N[:, None, :], axis=-1)/cm

    # calculate the apodization under the projection of the plane wave
    Oneg = np.array([ xele[0], 0])
    Opos = np.array([xele[-1], 0])
    Nperp = np.array([-np.cos(steers), np.sin(steers)]).T # rotate the normal vector + 90 degrees

    dX_OPneg = np.sum((P[None,:,:] - Oneg[None,None,:]) * Nperp[:,None,:], axis=-1)
    dX_OPpos = np.sum((P[None,:,:] - Opos[None,None,:]) * Nperp[:,None,:], axis=-1)
    apodtx = (dX_OPneg <= 0) & (dX_OPpos >= 0)

    return tautx, apodtx

def make_tabs_and_apods_2D(steers, r0, ctx, cm, xele, xout, zout, fnum):
    """Make delay tabs and apodizations for a 2D linear array with synthetic sources
    
    # Parameters
    - `steers`: steering angles in radians
    - `r0`: the radius of the synthetic source point in meters
        - a negative radius indicates a diverging wave
        - a radius of zero indicates a plane wave
        - a positive radius indicates a focused wave
    - `ctx`: assumed speed of spound on transmission in m/s
    - `cm`: speed of sound in the medium in m/s
    - `xele`: lateral position of the elements in meters
    - `xout`: lateral position of the output grid in meters
    - `zout`: axial position of the output grid in meters
    - `fnum`: the fnumber for recieve beamforming

    # Returns
    - `tautx`: the transmit delay tabs
    - `taurx`: the recieve delay tabs
    - `apodtx`: the transmit apodization
    - `apodrx`: the recieve apodization
    """

    import numpy as np

    if r0 > 0: 
        tautx, apodtx = calc_tx_tabs_and_apods_focused_2D(steers, r0, ctx, cm, xele, xout, zout)
    elif r0 == 0: 
        tautx, apodtx = calc_tx_tabs_and_apods_pw_2D(steers, ctx, cm, xele, xout, zout)
    else: 
        tautx, apodtx = calc_tx_tabs_and_apods_divergent_2D(steers, r0, ctx, cm, xele, xout, zout)

    taurx, apodrx = calc_rx_tabs_and_apods(cm, xele, xout, zout, fnum)

    return {"tautx":tautx, "taurx":taurx, "apodtx":apodtx, "apodrx":apodrx}
def calc_rx_synthetic_points(xele, fnum):
    """Calculates the receive synthetic point parameters for a linear array based on lateral element position and f-number

    # Parameters
    - `xele`: the lateral position of the elements in meters
    - `fnum`: the receive beamforming f-number

    # Returns
    - `ovecrx`: location of the receive sensors (nrx by 2)
    - `nvecrx`: normal vector of the recieve sensors (nrx by 2)
    - `alarx`: angular acceptance of the sensor - radians (nrx)
    """

    import numpy as np
    nrx = len(xele)
    ovecrx = np.array((xele, np.zeros(nrx))).T
    nvecrx = np.array((np.zeros(nrx), np.ones(nrx))).T
    alarx = np.arctan2(1/2,fnum) * np.ones(len(xele), dtype=np.float32)

    return ovecrx, nvecrx, alarx

def calc_tx_synthetic_points_pw_2D(steers, ctx, cm, xele, rpw:float=-10):
    """Make delay tabs and apodizations for a 2D linear array
    
    # Parameters
    - `steers`: steering angles in radians
    - `ctx`: assumed speed of spound on transmission in m/s
    - `cm`: speed of sound in the medium
    - `xele`: lateral position of the elements in meters
    - `rpw`: radius to approximate PW in meters

    # Returns
    - `ovectx`: the origin of the syntehtic point source
    - `nvectx`: the normal vector of the synthetic point source
    - `t0tx`: the time intercept of the synthetic point source
    - `doftx`: the depth of field around the DOF to use linear delay tabs
    - `alatx`: the angular aceptance reltive to the normal vector
    """

    import numpy as np

    # Change the angle of the plane waves to do global SOS correction 
    steers = np.arcsin(cm * np.sin(steers) / ctx)

    # point where the wave first crosses the aperture (set to be t=0)
    origtx = np.array([[xele[-1] if steer <= 0 else xele[0] for steer in steers], np.zeros(steers.shape)]).T

    # calculate normal vector of point source
    nvectx = np.array([np.sin(steers), np.cos(steers)]).T

    # calcualte point origins for the sources
    ovectx = rpw * nvectx

    # calculate the time at which the wave crosses the synthetic point source
    t0tx = np.sum(nvectx * (ovectx - origtx), axis=-1)/cm

    # depth of field around the point source to consider
    doftx = np.zeros(len(t0tx), dtype=np.float32)

    # calculate acceptance angle for plane wave source approximated as a point
    dxo = origtx - ovectx
    alatx = np.arccos(np.abs(np.sum(dxo * nvectx, axis=-1)) / np.linalg.norm(dxo, axis=-1))

    return ovectx, nvectx, t0tx, doftx, alatx

def calc_tx_synthetic_points_focused_2D(steers, ctx, cm, xele, rpw):
    """Make delay tabs and apodizations for a 2D linear array
    
    # Parameters
    - `steers`: steering angles in radians
    - `ctx`: assumed speed of spound on transmission in m/s
    - `cm`: speed of sound in the medium
    - `xele`: lateral position of the elements in meters
    - `r0`: radius fro origin of probe

    # Returns
    - `ovectx`: the origin of the syntehtic point source
    - `nvectx`: the normal vector of the synthetic point source
    - `t0tx`: the time intercept of the synthetic point source
    - `doftx`: the depth of field around the DOF to use linear delay tabs
    - `alatx`: the angular aceptance reltive to the normal vector
    """

    import numpy as np

    # Change the angle of the plane waves to do global SOS correction 
    if cm != ctx: raise NotImplementedError("Speed of sound correction has not yet been implemented for synthetic point datasets")

    # calcualte point origins for the sources
    ovectx = rpw * np.array([np.sin(steers), np.cos(steers)]).T

    # calcualte minimum distance from synthetic point to aperture
    dxmax = np.max(np.hypot(xele[None,:] - ovectx[:,0,None], ovectx[:,1,None]), axis=1)

    # calculate the time at which the wave crosses the synthetic point source
    t0tx = dxmax/cm

    alneg = np.atan2(ovectx[:,0] - xele[0], ovectx[:,1])
    alpos = np.atan2(ovectx[:,0] - xele[-1], ovectx[:,1])
    print(alneg, alpos)

    alatx = -(alpos - alneg)/2
    alsteers = (alpos + alneg)/2

    # calculate normal vector of point source
    nvectx = np.array([np.sin(alsteers), np.cos(alsteers)]).T

    # depth of field around the point source to consider
    doftx = np.zeros(len(t0tx), dtype=np.float32)

    return ovectx, nvectx, t0tx, doftx, alatx

def calc_tx_synthetic_points_divergent_2D(steers, ctx, cm, xele, rpw):
    """Make delay tabs and apodizations for a 2D linear array
    
    # Parameters
    - `steers`: steering angles in radians
    - `ctx`: assumed speed of spound on transmission in m/s
    - `cm`: speed of sound in the medium
    - `xele`: lateral position of the elements in meters
    - `r0`: radius fro origin of probe

    # Returns
    - `ovectx`: the origin of the syntehtic point source
    - `nvectx`: the normal vector of the synthetic point source
    - `t0tx`: the time intercept of the synthetic point source
    - `doftx`: the depth of field around the DOF to use linear delay tabs
    - `alatx`: the angular aceptance reltive to the normal vector
    """

    import numpy as np

    # Change the angle of the plane waves to do global SOS correction 
    if cm != ctx: raise NotImplementedError("Speed of sound correction has not yet been implemented for synthetic point datasets")

    # calcualte point origins for the sources
    ovectx = rpw * np.array([np.sin(steers), np.cos(steers)]).T

    # calcualte minimum distance from synthetic point to aperture
    dxmin = np.min(np.hypot(xele[None,:] - ovectx[:,0,None], ovectx[:,1,None]), axis=1)

    # calculate the time at which the wave crosses the synthetic point source
    t0tx = -dxmin/cm

    alneg = np.atan2(xele[0] - ovectx[:,0], -ovectx[:,1])
    alpos = np.atan2(xele[-1] - ovectx[:,0], -ovectx[:,1])

    alatx = (alpos - alneg)/2
    alsteers = (alpos + alneg)/2

    # calculate normal vector of point source
    nvectx = np.array([np.sin(alsteers), np.cos(alsteers)]).T

    # depth of field around the point source to consider
    doftx = np.zeros(len(t0tx), dtype=np.float32)

    return ovectx, nvectx, t0tx, doftx, alatx

def make_synthetic_points_2D(steers, r0, ctx, cm, xele, xout, zout, fnum):
    """Make delay tabs and apodizations for a 2D linear array with synthetic sources
    
    # Parameters
    - `steers`: steering angles in radians
    - `r0`: the radius of the synthetic source point in meters
        - a negative radius indicates a diverging wave
        - a radius of zero indicates a plane wave
        - a positive radius indicates a focused wave
    - `ctx`: assumed speed of spound on transmission in m/s
    - `cm`: speed of sound in the medium in m/s
    - `xele`: lateral position of the elements in meters
    - `xout`: the lateral pixel positions
    - `zout`: the axial pixel locations
    - `fnum`: the fnumber for recieve beamforming

    # Returns
    - `tautx`: the transmit delay tabs
    - `taurx`: the recieve delay tabs
    - `apodtx`: the transmit apodization
    - `apodrx`: the recieve apodization
    """

    import numpy as np

    Px, Pz = np.meshgrid(xout, zout, indexing='ij')
    pvec = np.ascontiguousarray(np.array([Px.flatten(), Pz.flatten()]).T)

    if  r0 == 0: ovectx, nvectx, t0tx, doftx, alatx = calc_tx_synthetic_points_pw_2D(steers, ctx, cm, xele)
    elif r0 < 0: ovectx, nvectx, t0tx, doftx, alatx = calc_tx_synthetic_points_divergent_2D(steers, ctx, cm, xele, r0)
    else:        ovectx, nvectx, t0tx, doftx, alatx = calc_tx_synthetic_points_focused_2D(steers, ctx, cm, xele, r0)

    ovecrx, nvecrx, alarx = calc_rx_synthetic_points(xele, fnum)

    params = dict(
        ovectx  = ovectx,
        nvectx  = nvectx, 
        t0tx    = t0tx, 
        doftx   = doftx, 
        alatx   = alatx,
        ovecrx  = ovecrx,
        nvecrx  = nvecrx,
        alarx   = alarx,
        c0      = cm,
        pnts    = pvec,
    )
    return params

def make_recon_grid_by_BW_2D(fnum, lam, xmin, xmax, zmin, zmax, BWx:float=1, BWz:float=2, x0is0:bool=True):
    """Make the reconstruction grid coordinates to meet 200% nyquist bandwidth sampling axially and laterally

    The sampling grids are set-up to scale with BW parameters.

    The lateral resolution `psfx` can be estimated from the fnumber and wavelength so that `psfx = fnum * lam`.
    The 100% bandwidth Nyquist sampling period `dx` can be calcualted so that `dx = psfx/2`. 
    Generalizing to a lateral bandwidth `BWx`, `dx` can be calculated so that `dx = psfx/(2*BWx)`.

    The axial spacing is determined by the 2-way wavelength `lam2w`. Given that `lam2w = lam/2`, the axial spacing `dz` can be calculated at 100% BW as `dz = lam2w/2 = lam/4`. 
    However, we often want a larger axial bandwidth `BWz` as `lam` should be the wavelength of the center frequency. 
    As such, we calcualte the axial sampling `dz` so that `dz = lam2w/(2*BWz)`
    
    # Parameters
    - `fnum`: F-number used in in beamforming - used to determine lateral resolution cell
    - `lam`: the imaging wavelength in meters
    - `xmin`: the minimum lateral coordinate (inclusive)
    - `xmax`: the maximun lateral coordinate (inclusive)
    - `zmin`: the minimum axial coordinate (inclusive)
    - `zmax`: the maximum axial coordinate (inclusive)
    - `BWx`: the lateral sampling bandwidth, lateral sampling at `fnum * lam / (2 * BWx)`
    - `BWz`: the axial sampling bandwidth, axial sampling is `(lam/2) / (2 * BWz)`
    - `x0is0`: indicates whether or not the x corrdinates are forced to include x = 0

    # Returns:
    - `xout`: a vector containing the lateral coordinates
    - `zout`: a vector containing the axial coordinates
    """

    import numpy as np

    # determine the axial and lateral spacing
    dz = lam / (4 * BWz) # 4 samples per two-way wavelength
    dx = lam * fnum / (2 * BWx) # 4 samples per lateral PSF

    zout = np.arange(zmin, zmax+dz/2, dz)

    if x0is0: # force sampling to be centered on x = 0
        ixmin = np.round(xmin/dx)
        ixmax = np.round(xmax/dx)
        nx = int(1 + ixmax - ixmin)
        xout = dx * (ixmin + np.arange(nx))

    else: 
        xout = np.arange(xmin, xmax+dx/2, dx)

    return xout, zout