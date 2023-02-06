import sys
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import os
from os.path import join as pjoin
import pybind11

CUDA = None
SOURCES = ["src/EvSpikeSimPackage.cpp",
           "src/LayersModule.cpp",
           "src/InitializersModule.cpp",
           "src/RandomModule.cpp"]

def locate_pybind_includes():
    return pybind11.get_include()

def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')
    self.src_extensions.append('.c')
    self.src_extensions.append('.cpp')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        self.set_executable('compiler_so', CUDA['nvcc'])
        # use only a subset of the extra_postargs, which are 1-1 translated
        # from the extra_compile_args in the Extension class
        postargs = extra_postargs['nvcc']

        if "--x=cu" not in cc_args:
            cc_args.append("--x=cu")

        super(obj, src, ext, cc_args, postargs, pp_opts)

    # inject our redefined _compile method into the class
    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


def setup_gpu(version):
    global CUDA
    CUDA = locate_cuda()

    ext = Extension('evspikesim',
                    sources=SOURCES,
                    library_dirs=["../build/lib/", CUDA['lib64']],
                    libraries=["evspikesim", "cuda", "cudart"],
                    runtime_library_dirs=[CUDA['lib64']],
                    # this syntax is specific to this build system
                    # we're only going to use certain compiler args with nvcc and not with gcc
                    # the implementation of this trick is in customize_compiler() below
                    extra_compile_args={'nvcc': ["-ccbin=g++", "--compiler-options", "-fPIC", "-std=c++17"]},
                    include_dirs=["../core/common/inc", "../core/gpu/inc", "./inc",
                                  locate_pybind_includes(), CUDA['include']])

    setup(name="EvSpikeSim",
          version=version,
          description="An Event-Based Spiking Neural Network Simulator written in C++",
          author="Florian Bacho",
          author_email="fb320@kent.ac.uk",
          ext_modules=[ext],
          cmdclass={'build_ext': custom_build_ext})


def setup_cpu(version):
    ext = Extension("evspikesim",
                    include_dirs=["../core/common/inc", "../core/cpu/inc", "./inc", locate_pybind_includes()],
                    library_dirs=["../build/lib"],
                    # Uncomment to use locally the built evspikesim library instead of installed version
                    libraries=["evspikesim"],
                    extra_compile_args=["-std=c++17"],
                    sources=SOURCES)

    setup(name="EvSpikeSim",
          version=version,
          description="An Event-Based Spiking Neural Network Simulator written in C++",
          author="Florian Bacho",
          author_email="fb320@kent.ac.uk",
          ext_modules=[ext])


if __name__ == "__main__":
    file = open("../version.txt")
    version = file.read()

    if "--gpu" in sys.argv:
        sys.argv.remove("--gpu")
        setup_gpu(version)
    else:
        setup_cpu(version)
