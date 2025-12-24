from setuptools import Extension, setup

# load requirements list
with open("requirements.txt", 'r') as f:
    reqs = f.readlines()

with open("req_depends.txt", 'r') as f:
    reqdeps = f.readlines()

# load the C extentsion library
trig = Extension(
    name="pycbf.trig._trig",
    include_dirs=["pycbf/trig"],
    depends=["pycbf/trig/trigengines.h"],
    sources=["pycbf/trig/trigengines.c"]
)

# run setup tools
setup(
    name='pycbf',
    description="C-Backed beamforming engines",
    author_email="wew12@duke.edu",
    packages=['pycbf', 'pycbf.trig'],
    install_requires=reqs,
    dependency_links=reqdeps,
    ext_modules=[trig],
    version="0.0.0"
)