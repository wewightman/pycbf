from setuptools import Extension, setup

# load the C extentsion library
trig = Extension(
    name="pycbf.trig._trig",
    include_dirs=["pycbf/trig"],
    depends=["pycbf/trig/trigengines.h"],
    sources=["pycbf/trig/trigengines.c"]
)

# run setup tools
setup(
    name='pyusel-pycbf',
    description="C-Backed beamforming engines",
    author_email="wew12@duke.edu",
    packages=['pycbf', 'pycbf.trig', 'pycbf.trig._trig'],
    package_dir={
        'pycbf':'pycbf', 
        'pycbf.trig':'pycbf/trig',
        'pycbf.trig._trig':'pycbf/trig'
    },
    license="MIT",
    ext_modules=[trig],
    version="0.0.0"
)