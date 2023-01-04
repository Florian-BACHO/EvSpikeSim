from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import sys
import numpy as np
import os
from os.path import join as pjoin

CUDA = None
CPU_SOURCES = ["src/python_interface/py_module_init.c",
               "src/python_interface/py_network.c",
               "src/python_interface/py_fc_layer.c",
               "src/python_interface/py_random.c",
               "src/spike_list.c",
               "src/random.c",
               "src/cpu/fc_layer.c",
               "src/fc_layer_params.c",
               "src/network.c"]

GPU_SOURCES = ["src/python_interface/py_module_init.c",
               "src/python_interface/py_network.c",
               "src/python_interface/py_fc_layer.c",
               "src/python_interface/py_random.c",
               "src/spike_list.c",
               "src/random.c",
               "src/gpu/fc_layer.cu",
               "src/fc_layer_params.c",
               "src/network.c"]

def setup_cpu():
    setup(name="EvSpikeSim",
          version="0.2.0",
          description="An Event-Based Spiking Neural Network Simulator written in C",
          author="Florian Bacho",
          author_email="fb320@kent.ac.uk",
          include_dirs = ["inc", np.get_include()],
          install_requires=["numpy"],
          ext_modules=[Extension("evspikesim", CPU_SOURCES)])

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
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

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

def setup_gpu():
    global CUDA
    CUDA = locate_cuda()
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()

    ext = Extension('evspikesim',
                sources=GPU_SOURCES,
                library_dirs=[CUDA['lib64']],
                libraries=['cuda', 'cudart', 'stdc++'],
                runtime_library_dirs=[CUDA['lib64']],
                # this syntax is specific to this build system
                # we're only going to use certain compiler args with nvcc and not with gcc
                # the implementation of this trick is in customize_compiler() below
                extra_compile_args={'gcc': [],
                                    'nvcc': ['-c', '--compiler-options', "'-fPIC'"]},
                include_dirs = ["inc", numpy_include, CUDA['include'], 'src'])
    
    setup(name="EvSpikeSim",
          version="0.1.1",
          description="An Event-Based Spiking Neural Network Simulator written in C",
          author="Florian Bacho",
          author_email="fb320@kent.ac.uk",
          install_requires=["numpy"],
          cmdclass={'build_ext': custom_build_ext},
          ext_modules=[ext])

if __name__ == "__main__":
    if "--gpu" in sys.argv:
        sys.argv.remove("--gpu")
        setup_gpu()
    setup_cpu()
