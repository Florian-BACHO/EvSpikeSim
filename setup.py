from distutils.core import setup, Extension

def main():
    setup(name="EvSpikeSim",
          version="0.1.0",
          description="An Event-Based Spiking Neural Network Simulator written in C",
          author="Florian Bacho",
          author_email="fb320@kent.ac.uk",
          include_dirs = ["inc"],
          install_requires=["numpy"],
          ext_modules=[Extension("evspikesim", ["src/python_interface/py_module_init.c",
                                               "src/python_interface/py_network.c",
                                               "src/python_interface/py_fc_layer.c",
                                               "src/python_interface/py_random.c",
                                               "src/spike_list.c",
                                               "src/random.c",
                                               "src/cpu/fc_layer.c",
                                               "src/fc_layer_params.c",
                                               "src/network.c"])])

if __name__ == "__main__":
    main()
