from setuptools import setup, Extension, find_packages

pycbfcpu = Extension(
    name="pycbf.cpu._pycbf",
    sources=["pycbf/cpu/pycbf.c"],
    depends=["pycbf/cpu/pycbf.h"],
    include_dirs=["pycbf/cpu"],  # keep local headers if you have any
)

if __name__ == "__main__":
    setup(
        name="pyusel-pycbf",
        version="1.1.0",
        description="C-backed beamforming engines",
        author_email="wew12@duke.edu",
        license="MIT",
        python_requires=">=3.10,<3.14",
        packages=find_packages(exclude=("tests", "examples", "benchmarks")),
        install_requires=[
            "numpy",
            "h5py",
        ],
        extras_require={
            "gpu": ["cupy"],
            "dev": ["pytest", "pytest-cov"],
        },
        package_data={
            "pycbf.gpu": ["__engines__.cu", "__advanced_engines__.cu"],
        },
        include_package_data=True,
        ext_modules=[pycbfcpu],
        zip_safe=False,
    )