//
// Created by Florian Bacho on 31/01/23.
//

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
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

static auto layer_get_weights(std::shared_ptr<Layer> layer) {
    auto &weights = layer->get_weights();
    auto shape = convert_dim(weights.get_dims());
    auto stride = get_strides(weights.get_dims());

    return py::array_t<float>(shape, stride, weights.get_c_ptr(), py::cast(layer));
}

static void layer_set_weights(std::shared_ptr<Layer> layer, const py::buffer &new_weights) {
    auto info = new_weights.request();
    const auto *new_weights_ptr = static_cast<float *>(info.ptr);
    auto &weights = layer->get_weights();

    weights.set_values(new_weights_ptr, new_weights_ptr + weights.size());
}

static auto layer_get_n_spikes(std::shared_ptr<Layer> layer) {
    auto &n_spikes = layer->get_n_spikes();
    std::vector<py::ssize_t> shape = {(py::ssize_t) n_spikes.size()};
    std::vector<py::ssize_t> stride = {sizeof(float)};

    return py::array_t<unsigned int>(shape, stride, n_spikes.data(), py::cast(layer));
}

static auto layer_get_synaptic_traces(std::shared_ptr<Layer> layer) {
    auto &synaptic_traces = layer->get_synaptic_traces();
    std::vector<unsigned int> dims = {layer->get_n_neurons(), layer->get_n_inputs(), layer->get_n_synaptic_traces()};
    auto shape = convert_dim(dims);
    auto stride = get_strides(dims);

    return py::array_t<float>(shape, stride, synaptic_traces.data(), py::cast(layer));
}

static auto layer_get_neuron_traces(std::shared_ptr<Layer> layer) {
    auto &neuron_traces = layer->get_neuron_traces();
    std::vector<unsigned int> dims = {layer->get_n_neurons(), layer->get_n_neuron_traces()};
    auto shape = convert_dim(dims);
    auto stride = get_strides(dims);

    return py::array_t<float>(shape, stride, neuron_traces.data(), py::cast(layer));
}

py::module create_layers_module(py::module &parent_module) {
    py::module layer_module = parent_module.def_submodule("layers", "A module containing the different layers of "
                                                                    "spiking neurons.");

    py::class_<Layer, std::shared_ptr<Layer>>(layer_module, "Layer", "Base class for layers.")
            .def_property_readonly("n_inputs", &Layer::get_n_inputs, "The number of input neurons.")
            .def_property_readonly("n_neurons", &Layer::get_n_neurons, "The number of neurons in the layer.")
            .def_property_readonly("tau_s", &Layer::get_tau_s, "The synaptic time constant.")
            .def_property_readonly("tau", &Layer::get_tau, "The membrane time constant (2 * tau_s).")
            .def_property_readonly("threshold", &Layer::get_threshold, "The threshold of the neurons.")
            .def_property("weights", &layer_get_weights, &layer_set_weights,
                          "A numpy ndarray containing the weights of the neuron")
            .def_property_readonly("post_spikes", py::cpp_function(&Layer::get_post_spikes,
                                                                   py::return_value_policy::reference_internal),
                                   "The post-synaptic spikes (updated during the inference).")
            .def_property_readonly("n_spikes", &layer_get_n_spikes,
                                   "The number of post-synaptic spikes fired by each neuron (updated during the"
                                   " inference).")
            .def_property_readonly("synaptic_traces", &layer_get_synaptic_traces,
                                   "A numpy ndarray containing the synaptic eligibility traces.")
            .def_property_readonly("neuron_traces", &layer_get_neuron_traces,
                                   "A numpy ndarray containing the neuron eligibility traces (to not be confused with "
                                   "synaptic traces)");

    py::class_<FCLayer, Layer, std::shared_ptr<FCLayer>>(layer_module, "FCLayer",
                                                         "A layer of Fully-Connected (FC) spiking neurons.");

    return layer_module;
}