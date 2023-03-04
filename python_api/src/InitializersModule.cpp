//
// Created by Florian Bacho on 31/01/23.
//

#include <evspikesim/Initializers/ConstantInitializer.h>
#include <evspikesim/Initializers/NormalInitializer.h>
#include <evspikesim/Initializers/UniformInitializer.h>
#include "InitializersModule.h"
#include "RandomModule.h"

namespace py = pybind11;
using namespace EvSpikeSim;

py::module create_initializers_module(py::module &parent_module) {
    py::module initializers_module = parent_module.def_submodule("initializers");

    initializers_module.doc() = "A module containing several weight initializers for Layer objects.\n\n";

    py::class_<Initializer, std::shared_ptr<Initializer>>(initializers_module, "Initializer",
                                                          "Interface for initializers.");

    py::class_<ConstantInitializer, Initializer, std::shared_ptr<ConstantInitializer>>
            (initializers_module, "ConstantInitializer", "Initializer that initializes with constant values.")
            .def(py::init<float>(), py::arg("value") = 0.0f, "Constructs the initializer with the given constant "
                                                             "value.\n\n"
                                                             ":param value: The constant used for initialization.\n"
                                                             ":type value: float");

    py::class_<NormalInitializer<RandomGenerator>, Initializer, std::shared_ptr<NormalInitializer<RandomGenerator>>>
            (initializers_module, "NormalInitializer", "Initializer that initializes with a normal distribution.")
            .def(py::init([](float mean, float stddev) {
                return NormalInitializer(global_random_generator, mean, stddev);
            }), py::arg("mean") = 0.0f, py::arg("stddev") = 1.0f, "Constructs the initializer with the given random "
                                                                  "generator, mean and standard deviation.\n\n"
                                                                  ":param mean: Mean of the normal distribution.\n"
                                                                  ":type mean: float\n"
                                                                  ":param stddev: Standard deviation of the normal "
                                                                  "distribution.\n"
                                                                  ":type stddev: float");

    py::class_<UniformInitializer<RandomGenerator>, Initializer, std::shared_ptr<UniformInitializer<RandomGenerator>>>
            (initializers_module, "UniformInitializer", "Initializer that initializes with a uniform distribution.")
            .def(py::init([](float lower_bound, float upper_bound) {
                     return UniformInitializer(global_random_generator, lower_bound, upper_bound);
                 }), py::arg("lower_bound") = -1.0f, py::arg("upper_bound") = 1.0f,
                 "Constructs the initializer with the given random generator, lower bound and upper bound.\n\n"
                 ":param lower_bound: Lower bound of the uniform distribution.\n"
                 ":type lower_bound: float\n"
                 ":param upper_bound: Upper bound of the uniform distribution.\n"
                 ":type upper_bound: float");

    return initializers_module;
}