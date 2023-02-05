//
// Created by Florian Bacho on 31/01/23.
//

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <evspikesim/Layers/LayerDescriptor.h>
#include <evspikesim/Layers/FCLayerDescriptor.h>
#include <evspikesim/Layers/Layer.h>
#include <evspikesim/Layers/FCLayer.h>
#include "LayersModule.h"

namespace py = pybind11;
using namespace EvSpikeSim;

/*
 * Layer Wrappers
 */

static auto convert_dim(const std::vector<unsigned int> &dims) {
    return std::vector<py::ssize_t>(dims.begin(), dims.end());
}

static auto get_strides(const std::vector<unsigned int> &dims) {
    std::vector<py::ssize_t> out(dims.size());
    py::ssize_t current_offset = sizeof(float);

    for (int i = dims.size() - 1; i >= 0; i--) {
        out[i] = current_offset;
        current_offset *= dims[i];
    }
    return out;
}

static auto layer_get_weights(Layer &layer) {
    auto &weights = layer.get_weights();
    auto shape = convert_dim(weights.get_dims());
    auto stride = get_strides(weights.get_dims());

    return py::array_t<float>(shape, stride, weights.c_ptr(), py::cast(layer));
}

static void layer_set_weights(Layer &layer, const py::buffer &new_weights) {
    auto info = new_weights.request();
    const auto *new_weights_ptr = static_cast<float *>(info.ptr);
    auto &weights = layer.get_weights();

    std::copy(new_weights_ptr, new_weights_ptr + weights.size(), weights.c_ptr());
}

py::module create_layers_module(py::module &parent_module) {
    py::module layer_module = parent_module.def_submodule("layers", "Layers and descriptors");

    py::class_<LayerDescriptor>(layer_module, "LayerDescriptor")
            .def(py::init<unsigned int, unsigned int, float, float>(),
                 py::arg("n_inputs"), py::arg("n_neurons"), py::arg("tau_s"), py::arg("threshold"))
            .def_readonly("n_inputs", &LayerDescriptor::n_inputs)
            .def_readonly("n_neurons", &LayerDescriptor::n_neurons)
            .def_readonly("tau_s", &LayerDescriptor::tau_s)
            .def_readonly("tau", &LayerDescriptor::tau)
            .def_readonly("threshold", &LayerDescriptor::threshold);

    py::class_<FCLayerDescriptor, LayerDescriptor>(layer_module, "FCLayerDescriptor")
            .def(py::init<unsigned int, unsigned int, float, float>(),
                 py::arg("n_inputs"), py::arg("n_neurons"), py::arg("tau_s"), py::arg("threshold"));

    py::class_<Layer, std::shared_ptr<Layer>>(layer_module, "Layer")
            .def_property_readonly("descriptor", py::cpp_function(&Layer::get_descriptor,
                                                                  py::return_value_policy::reference_internal))
            .def_property("weights", &layer_get_weights, &layer_set_weights)
            .def_property_readonly("post_spikes", py::cpp_function(&Layer::get_post_spikes,
                                                                   py::return_value_policy::reference_internal))
            .def_property_readonly("n_spikes", py::cpp_function(&Layer::get_n_spikes,
                                                                py::return_value_policy::reference_internal));

    py::class_<FCLayer, Layer, std::shared_ptr<FCLayer>>(layer_module, "FCLayer");

    return layer_module;
}