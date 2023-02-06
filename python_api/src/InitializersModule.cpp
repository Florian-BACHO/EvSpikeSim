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
    py::module initializers_module = parent_module.def_submodule("initializers", "Weights initializers");

    py::class_<Initializer, std::shared_ptr<Initializer>>(initializers_module, "Initializer");

    py::class_<ConstantInitializer, Initializer, std::shared_ptr<ConstantInitializer>>
    (initializers_module, "ConstantInitializer")
            .def(py::init<float>(), py::arg("value") = 0.0f);

    py::class_<NormalInitializer<RandomGenerator>, Initializer, std::shared_ptr<NormalInitializer<RandomGenerator>>>
            (initializers_module, "NormalInitializer")
            .def(py::init([](float mean, float stddev) {
                return NormalInitializer(global_random_generator, mean, stddev);
            }), py::arg("mean") = 0.0f, py::arg("stddev") = 1.0f);

    py::class_<UniformInitializer<RandomGenerator>, Initializer, std::shared_ptr<UniformInitializer<RandomGenerator>>>
            (initializers_module, "UniformInitializer")
            .def(py::init([](float lower_bound, float upper_bound) {
                return UniformInitializer(global_random_generator, lower_bound, upper_bound);
            }), py::arg("lower_bound") = -1.0f, py::arg("upper_bound") = 1.0f);

    return initializers_module;
}