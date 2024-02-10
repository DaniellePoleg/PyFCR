import os
import subprocess
import sys
import ninja
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DVERSION_INFO={__version__}"]

        ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
        cmake_args += [
            "-GNinja",
            f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
        ]

        build_args = []
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


exec(Path("pyfcr", "version.py").open().read())
setup(
    name='PyFCR',
    version=__version__,
    author="Danielle Poleg",
    author_email="daniellepoleg@gmail.com",
    description="A Python package for frailty-based multivariate survival data analysis with competing risks",
    long_description='PyFCR is a robust Python package used for survival estimation procedure under the multivariate normal frailty model. It accommodates any number of competing risks and diverse configurations.',
    ext_modules=[CMakeExtension(name="pyfcr._estimators", sourcedir=str(Path("pyfcr") / "estimators"))],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=find_packages(),
    python_requires="~=3.10"
)
