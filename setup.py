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
    packages=['pycbf', 'pycbf.cpu', 'pycbf.cpu._pycbf', 'pycbf.gpu', 'pycbf.any', 'pycbf.dataio', 'pycbf.helpers'],
    package_dir={
        'pycbf':'pycbf', 
        'pycbf.cpu':'pycbf/cpu',
        'pycbf.cpu._pycbf':'pycbf/cpu',
        'pycbf.gpu':'pycbf/gpu',
        'pycbf.any':'pycbf/any',
        'pycbf.dataio':'pycbf/dataio',
        'pycbf.helpers':'pycbf/helpers'
    },
    license="MIT",
    python_requires='>=3.10,<3.14',
    package_data={'pycbf.gpu':["__engines__.cu", "__advanced_engines__.cu"]},
    ext_modules=[pycbfcpu],
    version="1.1.0",
    requires=[
        "numpy",
        "h5py",
        "cupy"
    ]
)