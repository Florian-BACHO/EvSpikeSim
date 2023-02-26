//
// Created by Florian Bacho on 28/01/23.
//

#define PYBIND11_DETAILED_ERROR_MESSAGES

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
    m.doc() = "\n.. currentmodule:: evspikesim\n\n"
              "evspikesim\n"
              "==========\n\n"
              "The main module of EvSpikeSim.\n\n"
              ".. rubric:: Modules\n\n"
              ".. autosummary::\n"
              "   :toctree: _generate\n\n"
              "   initializers\n"
              "   layers\n"
              "   random\n\n"
              ".. rubric:: Classes\n\n"
              ".. autosummary::\n"
              "   :toctree: _generate\n\n"
              "   Spike\n"
              "   SpikeArray\n"
              "   SpikingNetwork\n\n";

    py::class_<Spike>(m, "Spike", "Spike event. Spikes are fired and received by spiking neurons. They drive the "
                                  "computation during inference.")
            .def(py::init<unsigned int, float>(), py::arg("index"), py::arg("time"),
                 "Constructs a spike with a given index and timing.\n\n"
                 ":param index: The index of the neuron that fired the spike.\n"
                 ":type index: int\n"
                 ":param time: The timing of the spike.\n"
                 ":type time: float\n")
            .def_readwrite("index", &Spike::index, "The index of the neuron that fired the spike.")
            .def_readwrite("time", &Spike::time, "The timing of the spike.")
            .def(py::self == py::self, "Checks if two spikes occured within a time distance of 1e-6, "
                                       "i.e. abs(self.time - arg0.time) < 1e-6.\n\n"
                                       ":param arg0: The other spike to compare.\n"
                                       ":type arg0: Spike\n"
                                       ":return: True if the absolute difference between the two spike timings is "
                                       "bellow 1e-6.\n"
                                       ":rtype: bool")
            .def(py::self != py::self, "Checks if two spikes did not occur within a time distance of 1e-6, "
                                       "i.e. abs(self.time - arg0.time) >= 1e-6.\n\n"
                                       ":param arg0: The other spike to compare.\n"
                                       ":type arg0: Spike\n"
                                       ":return: True if the absolute difference between the two spike timings is above "
                                       "1e-6.\n"
                                       ":rtype: bool")
            .def(py::self < py::self,
                 "Checks if the spike occured before a given spike, i.e. self.time < arg0.time.\n\n"
                 ":param arg0: The other spike to compare.\n"
                 ":type arg0: Spike\n"
                 ":return: True if the spike occured before arg0.\n"
                 ":rtype: bool")
            .def(py::self <= py::self, "Checks if the spike occured before or at the same time as given spike, "
                                       "i.e. self.time <= arg0.time.\n\n"
                                       ":param arg0: The other spike to compare.\n"
                                       ":type arg0: Spike\n"
                                       ":return: True if the spike occured before or at the same time as arg0.\n"
                                       ":rtype: bool")
            .def(py::self > py::self, "Checks if the spike occured after a given spike, i.e. self.time > arg0.time.\n\n"
                                      ":param arg0: The other spike to compare.\n"
                                      ":type arg0: Spike\n"
                                      ":return: True if the spike occured after arg0.\n"
                                      ":rtype: bool")
            .def(py::self >= py::self, "Checks if the spike occured after or at the same time as given spike, "
                                       "i.e. self.time >= arg0.time.\n\n"
                                       ":param arg0: The other spike to compare.\n"
                                       ":type arg0: Spike\n"
                                       ":return: True if the spike occured after or at the same time as arg0.\n"
                                       ":rtype: bool")
            .def("__repr__", &obj_to_str<Spike>, "Formats the spike to a string.\n\n"
                                                 ":return: A formated string describing the spike.\n"
                                                 ":rtype: str");

    py::class_<SpikeArray>(m, "SpikeArray", "An array of Spike. SpikeArray objects must be sorted in time before being "
                                            "used for inference.")
            .def(py::init(), "Constructs an empty array of spikes.")
            .def(py::init<const std::vector<unsigned int> &, const std::vector<float> &>(), py::arg("indices"),
                 py::arg("times"), "Constructs an array of spikes with the given spike indices and spike timings.\n\n"
                                   ":param indices: Indices of spikes.\n"
                                   ":type indices: list[int]\n"
                                   ":param times: Timings of spikes.\n"
                                   ":type times: list[float]")
            .def(py::init(&spike_array_buffer_init), py::arg("indices"), py::arg("times"),
                 "Constructs an array of spikes with the given spike indices and "
                 "spike timings.\n\n"
                 ":param indices: Indices of spikes. The buffer can be a numpy ndarray. "
                 "The data type of the buffer must be uint32.\n"
                 ":type indices: buffer\n"
                 ":param times: Timings of spikes. The buffer can be a numpy ndarray. "
                 "The data type of the buffer must be float32.\n"
                 ":type times: buffer")
            .def("add", static_cast<void (SpikeArray::*)(unsigned int, float)>(&SpikeArray::add),
                 py::arg("index"), py::arg("time"), "Adds a single spike to the array.\n\n"
                                                    ":param index: The index of the neuron that fired the spike.\n"
                                                    ":type index: int\n"
                                                    ":param time: The timing of the spike.\n"
                                                    ":type time: float\n")
            .def("add", static_cast<void (SpikeArray::*)(const std::vector<unsigned int> &,
                                                         const std::vector<float> &)>(&SpikeArray::add),
                 py::arg("indices"), py::arg("times"), "Adds several spikes to the array given indices and timings.\n\n"
                                                       ":param indices: Neuron indices that fired the spikes.\n"
                                                       ":type indices: list[int]\n"
                                                       ":param times: Timings of the spikes.\n"
                                                       ":type times: list[float]")
            .def("sort", &SpikeArray::sort, "Sorts the spike array in time. Must be called before being used for "
                                            "inference.")
            .def("is_empty", &SpikeArray::is_empty, "Checks if the spike array is empty.\n\n"
                                                    ":return: True if the array is empty.\n"
                                                    ":rtype: bool")
            .def("clear", &SpikeArray::clear, "Empties the spike array.")
            .def("__len__", &SpikeArray::n_spikes, "Gets the number of spikes in the array.\n\n"
                                                   ":return: The size of the array.\n"
                                                   ":rtype: int")
            .def("__iter__", &get_obj_iterator<SpikeArray>, py::keep_alive<0, 1>(),
                 "Gets a constant iterator on the first spike.\n\n"
                 ":return: A constant iterator on the first spike.\n"
                 ":rtype: Iterator")
            .def("__repr__", &obj_to_str<SpikeArray>, "Formats the spike array to the given output stream.\n\n"
                                                      ":return: A string containing all formated spikes.\n"
                                                      ":rtype: str")
            .def(py::self == py::self, "Checks the equality between two spike arrays.\n\n"
                                       ":param arg0: The other spike array to compare.\n"
                                       ":type arg0: SpikeArray\n"
                                       ":return: True if the two arrays have the same number of spikes and if all the "
                                       "spikes are equals.\n"
                                       ":rtype: bool")
            .def(py::self != py::self, "Checks the equality between two spike arrays.\n\n"
                                       ":param arg0: The other spike array to compare.\n"
                                       ":type arg0: SpikeArray\n"
                                       ":return: True if the two arrays do not have the same number of spikes or if at "
                                       "least one spike is not equal to the corresponding one in rhs.\n"
                                       ":rtype: bool")
            .def_property_readonly("n_spikes", &SpikeArray::n_spikes, "The number of spikes in the array.");

    py::class_<SpikingNetwork>(m, "SpikingNetwork", "Spiking Neural Network (SNN) composed of layers of spiking "
                                                    "neurons.")
            .def(py::init<const std::string &>(), py::arg("compile_path") = SpikingNetwork::default_compile_path,
                 "Constructs an empty SNN.\n\n"
                 ":param compile_path: Compilation path for custom kernel callback sources.\n"
                 ":type compile_path: str\n")
            .def("add_fc_layer", static_cast<std::shared_ptr<FCLayer> (SpikingNetwork::*)(unsigned int,
                                                                                          unsigned int,
                                                                                          float,
                                                                                          float,
                                                                                          Initializer &,
                                                                                          unsigned int)>
                 (&SpikingNetwork::add_layer),
                 py::arg("n_inputs"), py::arg("n_neurons"), py::arg("tau_s"), py::arg("threshold"),
                 py::arg("initializer"), py::arg("buffer_size") = 64u,
                 "Adds a fully-connected (FC) layer with the given arguments to the network. "
                 "The added layer uses the default kernel during inference.\n\n"
                 ":param n_inputs: The number of input neurons.\n"
                 ":type n_inputs: int\n"
                 ":param n_neurons: The number of neurons in the layer.\n"
                 ":type n_neurons: int\n"
                 ":param tau_s: The synaptic time constant of the neurons. The membrane time constant will be "
                 "defined as twice tau_s.\n"
                 ":type tau_s: float\n"
                 ":param threshold: The threshold of the neurons.\n"
                 ":type threshold: float\n"
                 ":param initializer: The initializer to use to initialize the weights of the layer.\n"
                 ":type initializer: Initializer.\n"
                 ":param buffer_size: The size (per neuron) of the post-synaptic spike times buffer used during "
                 "inference.\n"
                 ":type buffer°size: int\n"
                 ":return: The newly created FCLayer object.\n"
                 ":rtype: FCLayer\n")
            .def("add_fc_layer_from_source",
                 static_cast<std::shared_ptr<FCLayer> (SpikingNetwork::*)(const std::string &,
                                                                          unsigned int,
                                                                          unsigned int,
                                                                          float,
                                                                          float,
                                                                          Initializer &,
                                                                          unsigned int)>
                 (&SpikingNetwork::add_layer_from_source), py::arg("n_inputs"), py::arg("src_path"),
                 py::arg("n_neurons"), py::arg("tau_s"), py::arg("threshold"), py::arg("initializer"),
                 py::arg("buffer_size") = 64u,
                 "Adds a layer of type LayerType with a custom kernel callbacks source. "
                 "If not already used, the given source file is compiled and loaded by the JITCompiler.\n\n"
                 ":param src_path: Path to the kernel callbacks source file.\n"
                 ":type src_path: str\n"
                 ":param n_inputs: The number of input neurons.\n"
                 ":type n_inputs: int\n"
                 ":param n_neurons: The number of neurons in the layer.\n"
                 ":type n_neurons: int\n"
                 ":param tau_s: The synaptic time constant of the neurons. The membrane time constant will be "
                 "defined as twice tau_s.\n"
                 ":type tau_s: float\n"
                 ":param threshold: The threshold of the neurons.\n"
                 ":type threshold: float\n"
                 ":param initializer: The initializer to use to initialize the weights of the layer.\n"
                 ":type initializer: Initializer.\n"
                 ":param buffer_size: The size (per neuron) of the post-synaptic spike times buffer used during "
                 "inference.\n"
                 ":type buffer°size: int\n"
                 ":return: The newly created FCLayer object.\n"
                 ":rtype: FCLayer\n")
            .def("infer", static_cast<const SpikeArray &(SpikingNetwork::*)(const SpikeArray &)>
                 (&SpikingNetwork::infer), py::arg("inputs"),
                 "Infer the entire SNN using the given input spike array.\n\n"
                 ":param inputs: Sorted input spikes.\n"
                 ":type inputs: SpikeArray.\n"
                 ":return: The output spike array.\n"
                 ":rtype: SpikeArray\n")
            .def("infer", static_cast<const SpikeArray &(SpikingNetwork::*)(const std::vector<unsigned int> &,
                                                                            const std::vector<float> &)>
                 (&SpikingNetwork::infer), py::arg("indices"), py::arg("times"),
                 "Infer the entire SNN using the given input spike indices and times.\n\n"
                 ":param indices: Input spike indices.\n"
                 ":type indices: List[int]\n"
                 ":param times: Input spike times.\n"
                 ":type times: List[float]\n"
                 ":return: The output spike array.\n"
                 ":rtype: SpikeArray\n")
            .def("__len__", &SpikingNetwork::get_n_layers,
                 "Gets the number of layers.\n\n"
                 ":return: The number of layers.\n"
                 ":rtype: int\n")
            .def("__iter__", &get_obj_iterator<SpikingNetwork>, py::keep_alive<0, 1>(),
                 "Gets an iterator on the first layer.\n\n"
                 ":return: An iterator on the first layer.\n"
                 ":rtype: Iterator")
            .def("__getitem__", &SpikingNetwork::operator[] < unsigned int > , py::arg("index"),
                 "Gets the layer at the specified index.\n\n"
                 ":param index: The index of the layer.\n"
                 ":type index: int\n"
                 ":return: The requested layer.\n"
                 ":rtype: Layer\n")
            .def_property_readonly("output_layer", &SpikingNetwork::get_output_layer,
                                   "The output layer (i.e. the last layer of the network).");
}

PYBIND11_MODULE(evspikesim, m
) {
create_main_module(m);
create_layers_module(m);
create_initializers_module(m);
create_random_module(m);
}