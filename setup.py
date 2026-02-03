from pathlib import Path
import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

def shared_lib_name():
    if   sys.platform.startswith('win'):
        return "pycbf.dll"
    elif sys.platform.startswith('linux'):
        return "pycbf.so"
    elif sys.platform == "darwin":
        return "libpycbf.dylib"
    else:
        raise Exception(f"OS platform must be one windows, linux, or mac (darwin), but was {sys.platform}")

class build_shared_ext(_build_ext):
    """
    Build a plain shared library (NOT a Python extension) and place it
    into build_lib/pycbf/cpu so it is included in wheels.
    """

    def build_extension(self, ext):
        # setuptools-defined build directories
        build_cmd = self.get_finalized_command("build")
        build_lib = Path(build_cmd.build_lib)
        build_temp = Path(self.build_temp)

        # target location inside build output
        target_dir = build_lib / "pycbf" / "cpu"
        target_dir.mkdir(parents=True, exist_ok=True)
        build_temp.mkdir(parents=True, exist_ok=True)

        # compile relative source paths → object files
        objects = self.compiler.compile(
            ext.sources,
            output_dir=str(build_temp),
        )

        # link shared library into build_lib package directory
        outname = target_dir / shared_lib_name()
        self.compiler.link_shared_object(
            objects,
            outname.name,
            output_dir=str(outname.parent),
        )

        print(f"Built shared library: {outname}")

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
        package_data={
            "pycbf.gpu": ["__engines__.cu", "__advanced_engines__.cu"],
        },
        include_package_data=True,
        ext_modules=[pycbfcpu],
        cmdclass={"build_ext":build_shared_ext},
        zip_safe=False,
    )