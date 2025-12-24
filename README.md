# pycbf
This repository contains extension accelerated beamformers to be used with Python. My goal with this repository is to make an easily interpretable, easy-to-use, flexible, and decently performent set of software beaformers that can be used on any (almost) system.

My current research area of focus is ultrasound shear wave elastography imaging, so many examples will be relevant to that field.

## Compilation and Installation
These scripts should be installable on all systems with pip install - though if installing on Windows - you will need the C compilation toolkits to install automatically. It is a goal of mine to pre-compile windows binaries in the future and make this repo available on pipy - but that is currently out of the scope of my work.

```
git clone https://github.com/wewightman/pycbf.git
cd pycbf
python -m pip install .
```

### GPU functionality
All GPU functionality is achieved though the use of CuPy - which is best tested and validated for Nvidia GPUs - though future releases will also support other GPU types.

## Background and philosophy
### Basics of Delay-And-Sum (DAS) beamforming
Beamforming is an essential tool in many fields, especially so in the field of medical ultrasound imaging.
All algorithms in this repository are - at their core - based on delay-and-sum (DAS) beamforming, with an excellent introduction found in [this publication](https://www.sciencedirect.com/science/article/pii/S0041624X20302444) [1].

In short, medical ultrasound imaging works by transmitting and receiving pressure signals from pressure transmitters (only a signal source), pressure transcievers (only a signal sensor), or pressure transducers (both signal sources and sensors). 
Slight differences in the mechanical properties of the tissue lead to reflections of the pressure waves that are sent out from the transducer, reflections that can be sensed by each of the independent elements in the aperture.
Multiple elements can be combined to form an aperture, or a set of transmitting sources and receiving sensors.

To triangulate the source of the reflection - you can use an assumed speed of sound in tissue $c$ - usually assumed to be ~ 1540 m/s in soft tissue. 
Using this assumed speed of sound, you can calculate how long the pressure wave took to travel from the source point $\vec{x}_{tx}$, the field point $\vec{x}_f$, and the receive point $\vec{x}_{rx}$.
The travel time for the full path $t_f$ can be represented as the sum of the transmit travel time $\tau_{tx}$ from $\vec{x}_{tx}$ to the field point $\vec{x}_f$ and the receive travel time $\tau_{tx}$ from $\vec{x}_f$ to the field point $\vec{x}_{rx}$. 
Assuming that the source, scatter, and sensor are all point like, 

$$t_f = \tau_{tx} + \tau_{rx} = \|\vec{x}_f - \vec{x}_s\|/c + \|\vec{x}_r - \vec{x}_f\|/c.$$

The delayed pressure signal at $p(t_f)$ can then be extracted to find the magnitude of the reflection at the point $\vec{x}_f$.
To improve the true estimate of the reflected signal at the point $\vec{x}_f$, multiple estimates of $p_{tx,rx}(t_f)$ from all transmit-receive pairs can be summed together to get a DAS image.
You can also take the weighted sum of the signals from all the transmit-receive pairs, with unique transmit apodization $a_{tx}(\vec{x}_f)$ and receive apodization $a_{rx}(\vec{x}_f)$ so that the reconstucted pressure field $P$ takes the form

$$P(\vec{x}_f) = \sum_{tx,rx} a_{tx}(\vec{x}_f) \cdot a_{rx}(\vec{x}_f) \cdot p_{tx,rx}(t_f).$$

This approach highlights that beamforming can be done on a truly pixel-by-pixel basis. This pixel-by-pixel approach then has two steps:
 1. Calculate $\tau_{tx}$, $\tau_{rx}$, $a_{tx}$, and $a_{rx}$ for each $\vec{x}_f$
 2. Interpolate $p_{tx,rx}$ at time $t_f$.

All beamformers in this repository achieve this point like approach using one of two approaches: A tabbed approach, and a synthetic point approach. As all beamformers take a pythonic class-based form, all beamforming objects are either of the class `Tabbed` or `SynthPointed`.
These approaches have been implemented and tested using compiled C kernels for both the CPU and GPU.

### `Tabbed` software beamforming class
The software beamforming classes that inherit from the `Tabbed` class require the user to input/precalculate all transmit and receive delay tabs and apodizations.
The minimum inputs for a `Tabbed` beamformer are four matrices: `tautx`, `taurx`, `apodtx`, and `apodrx`. 
If `Np` is the number of beamforming points, `Ntx` is the number of transmit events, and `Nrx` is the number of receive events...
 - `tautx` and `apodtx` both have the shape `Ntx` by `Np`
 - `taurx` and `apodrx` both have the shape `Nrx` by `Np`. 

These software beamformers are, essentially, specialized interpolation objects.
This general purpose class of beamformer allows for any variation of delay tab calculation - including delay tab calculation for arbitrary speed of sound maps.
Future releases will include easy wrapper functions to generate these four matrices for common imaging configurations.

### `SynthPointed` software beamforming class
The software beamforming classes that inherit from the `SynthPointed` class require more strict assumptions, but allow for very rapid GPU-based beamforming as delay tab calculation is simplified and requires less global memory access during run time.
This approach is based on approximating all signal sources and signal sensors as points in space with some position and directivity.

This class requires an assumed speed of sound, position, orientation, and angular sensitivity of all transmit and receive virtual points, as well as the time at which the wave field passes through the transmit virtual points.

This approach is currently not implemented and not recommended for the CPU-based classes as the extra repeated computations for the delay tabs could be prohibitive. Future releases will have this approach implemented on the CPU to allow testing of either algorithm on any system regardless of the available resources.

## References
[1] V. Perrot, M. Polichetti, F. Varray, and D. Garcia, “So you think you can DAS? A viewpoint on delay-and-sum beamforming,” Ultrasonics, vol. 111, p. 106309, Mar. 2021, doi: 10.1016/j.ultras.2020.106309.

## Testing
Unit testing has not yet been implemented - however, you may run the example scripts.

## Contributors
 - @wewightman: Owner and sole developer

