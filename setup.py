from setuptools import Extension, setup

# load the C extentsion library
pycbfcpu = Extension(
    name="pycbf.cpu._pycbf",
    include_dirs=["pycbf/cpu/"],
    depends=["pycbf/cpu/pycbf.h"],
    sources=["pycbf/cpu/pycbf.c"]
)

# run setup tools
setup(
    name='pyusel-pycbf',
    description="C-Backed beamforming engines",
    author_email="wew12@duke.edu",
    packages=['pycbf', 'pycbf.cpu', 'pycbf.cpu._pycbf', 'pycbf.gpu'],
    package_dir={
        'pycbf':'pycbf', 
        'pycbf.cpu':'pycbf/cpu',
        'pycbf.cpu._pycbf':'pycbf/cpu',
        'pycbf.gpu':'pycbf/gpu'
    },
    license="MIT",
    package_data={'pycbf.gpu':["__engine__.cu"]},
    ext_modules=[pycbfcpu],
    version="1.0.0",
    requires=[
        "numpy"
    ]
)