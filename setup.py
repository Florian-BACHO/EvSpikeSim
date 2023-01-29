from distutils.core import setup, Extension

CPU_SOURCES = ["python_api/src/EvSpikeSimPackage.cpp",
               "python_api/src/LayersModule.cpp"]
BOOST_LIB_DIR = "/usr/local/opt/boost-python3/lib"

if __name__ == "__main__":
    file = open("version.txt")
    version = file.read()

    cpu_extension = Extension("evspikesim",
                              include_dirs = ["inc", "python_api/inc"],
                              library_dirs = [BOOST_LIB_DIR, "build/lib/"],
                              libraries=["boost_python311", "boost_numpy311", "evspikesim"],
                              extra_compile_args=['-std=c++20'],
                              sources=CPU_SOURCES)
    setup(name="EvSpikeSim",
          version=version,
          description="An Event-Based Spiking Neural Network Simulator written in C",
          author="Florian Bacho",
          author_email="fb320@kent.ac.uk",
          ext_modules=[cpu_extension])
