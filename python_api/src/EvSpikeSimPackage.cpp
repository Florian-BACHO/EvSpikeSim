//
// Created by Florian Bacho on 28/01/23.
//

#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <evspikesim/Spike.h>
#include <evspikesim/SpikeArray.h>
#include <evspikesim/SpikingNetwork.h>
#include <evspikesim/Initializers/Initializer.h>
#include "LayersModule.h"
#include "InitializersModule.h"
#include "RandomModule.h"

namespace py = pybind11;
using namespace EvSpikeSim;

/*
 * Generic Wrappers
 */
template<class T>
static std::string obj_to_str(const T &obj) {
    std::stringstream stream;

    stream << obj;
    return stream.str();
}

template<class T>
static py::iterator get_obj_iterator(T &obj) {
    return py::make_iterator(obj.begin(), obj.end());
}

/*
 * SpikeArray Wrappers
 */

static SpikeArray spike_array_buffer_init(py::buffer &indices, py::buffer &times) {
    auto indices_info = indices.request();
    auto times_info = times.request();
    auto size = indices_info.shape[0];
    auto indices_ptr = static_cast<unsigned int *>(indices_info.ptr);
    auto times_ptr = static_cast<float *>(times_info.ptr);

    return SpikeArray(indices_ptr, indices_ptr + size, times_ptr);
}

static void create_main_module(py::module &m) {
    m.doc() = "An Event-Based Spiking Neural Network Simulator written in C++";

    py::class_<Spike>(m, "Spike")
            .def(py::init<unsigned int, float>())
            .def_readwrite("index", &Spike::index)
            .def_readwrite("time", &Spike::time)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def(py::self < py::self)
            .def(py::self <= py::self)
            .def(py::self > py::self)
            .def(py::self >= py::self)
            .def("__repr__", &obj_to_str<Spike>);

    py::class_<SpikeArray>(m, "SpikeArray")
            .def(py::init())
            .def(py::init<const std::vector<unsigned int> &, const std::vector<float> &>(),
                 py::arg("indices"), py::arg("times"))
            .def(py::init(&spike_array_buffer_init),
                 py::arg("indices"), py::arg("times"))
            .def("add", static_cast<void (SpikeArray::*)(unsigned int, float)>(&SpikeArray::add),
                 py::arg("index"), py::arg("time"))
            .def("add", static_cast<void (SpikeArray::*)(const std::vector<unsigned int> &,
                                                         const std::vector<float> &)>(&SpikeArray::add),
                 py::arg("indices"), py::arg("times"))
            .def("sort", &SpikeArray::sort)
            .def("empty", &SpikeArray::is_empty)
            .def("clear", &SpikeArray::clear)
            .def("__len__", &SpikeArray::n_spikes)
            .def("__iter__", &get_obj_iterator<SpikeArray>, py::keep_alive<0, 1>())
            .def("__repr__", &obj_to_str<SpikeArray>)
            .def(py::self == py::self)
            .def(py::self != py::self)
            .def_property_readonly("n_spikes", &SpikeArray::n_spikes);

    py::class_<SpikingNetwork>(m, "SpikingNetwork")
            .def(py::init())
            .def("add_layer", static_cast<std::shared_ptr<FCLayer> (SpikingNetwork::*)(const FCLayerDescriptor &,
                                                                                       unsigned int)>
            (&SpikingNetwork::add_layer), py::arg("descriptor"), py::arg("buffer_size") = 64u)
            .def("add_layer", static_cast<std::shared_ptr<FCLayer> (SpikingNetwork::*)(const FCLayerDescriptor &,
                                                                                       Initializer &,
                                                                                       unsigned int)>
            (&SpikingNetwork::add_layer), py::arg("descriptor"), py::arg("initializer"), py::arg("buffer_size") = 64u)
            .def("infer", static_cast<const SpikeArray &(SpikingNetwork::*)(const SpikeArray &)>
            (&SpikingNetwork::infer), py::arg("inputs"))
            .def("infer", static_cast<const SpikeArray &(SpikingNetwork::*)(const std::vector<unsigned int> &,
                                                                            const std::vector<float> &)>
                 (&SpikingNetwork::infer),
                 py::arg("indices"), py::arg("times"))
            .def("__len__", &SpikingNetwork::get_n_layers)
            .def("__iter__", &get_obj_iterator<SpikingNetwork>, py::keep_alive<0, 1>())
            .def("__getitem__", &SpikingNetwork::operator[] < unsigned int > )
            .def_property_readonly("output_layer", &SpikingNetwork::get_output_layer);
}

PYBIND11_MODULE(evspikesim, m
) {
create_main_module(m);
create_layers_module(m);
create_initializers_module(m);
create_random_module(m);
}