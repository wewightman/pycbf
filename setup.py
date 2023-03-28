from setuptools import Extension, setup

trig = Extension(
    name="pycbf.trig._trig",
    include_dirs=["pycbf/trig"],
    depends=["pycbf/trig/trigengines.h"],
    sources=["pycbf/trig/trigengines.c"]
)

setup(
    name='pycbf',
    description="C-Backed beamforming engines",
    author_email="wew12@duke.edu",
    packages=['pycbf', 'pycbf.trig'],
    ext_modules=[trig]
)