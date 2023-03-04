//
// Created by Florian Bacho on 31/01/23.
//

#include "RandomModule.h"

namespace py = pybind11;
using namespace EvSpikeSim;

RandomGenerator global_random_generator = RandomGenerator();

void set_seed(unsigned long seed) {
    global_random_generator.seed(seed);
}

py::module create_random_module(py::module &parent_module) {
    py::module random_module = parent_module.def_submodule("random", "Global random generator module used by "
                                                                     "initializers. By default, the random generator "
                                                                     "uses the timestamp as initial seed.");

    random_module.def("set_seed", &set_seed, py::arg("seed"),
                      "Sets the seed of the random generator.\n\n"
                      ":param seed: The new seed of the random generator.\n"
                      ":type seed: int");

    return random_module;
}