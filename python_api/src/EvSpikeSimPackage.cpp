//
// Created by Florian Bacho on 28/01/23.
//

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "Spike.h"
#include "SpikeArray.h"
#include "LayersModule.h"
#include "SpikingNetwork.h"

using namespace boost::python;
using namespace EvSpikeSim;
namespace np = boost::python::numpy;

/*
 * SpikeArray wrappers
 */

static void check_indices_times_np_arrays(const np::ndarray &indices, const np::ndarray &times) {
    // Check arrays data types
    if (indices.get_dtype() != np::dtype::get_builtin<unsigned int>() ||
        times.get_dtype() != np::dtype::get_builtin<float>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        throw_error_already_set();
    }

    // Check number of dimensions
    if (indices.get_nd() != 1 || times.get_nd() != 1) {
        PyErr_SetString(PyExc_TypeError, "Number of dimensions for indices and times arrays should be 1");
        throw_error_already_set();
    }

    // Check number of indices and times
    if (indices.shape(0) != times.shape(0)) {
        PyErr_SetString(PyExc_TypeError, "Number of indices does not match the number of times");
        throw_error_already_set();
    }
}

static boost::shared_ptr<SpikeArray> spike_array_init(object indices, object times) {
    if (indices.is_none() || times.is_none()) {
        return boost::shared_ptr<SpikeArray>(new SpikeArray());
    }

    extract<np::ndarray> indices_extract(indices);
    extract<np::ndarray> times_extract(times);

    if (!indices_extract.check() || !times_extract.check()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect argument type. It should be numpy arrays");
        throw_error_already_set();
    }

    auto np_indices = indices_extract();
    auto np_times = times_extract();

    check_indices_times_np_arrays(np_indices, np_times);

    // Valid arguments -> instanciate new SpikeArray object
    const auto *indices_start = (unsigned int *) np_indices.get_data();
    const auto *indices_end = indices_start + np_indices.shape(0);
    const auto *times_start = (float *) np_times.get_data();

    return boost::shared_ptr<SpikeArray>(new SpikeArray(indices_start, indices_end, times_start));
}

static void (SpikeArray::*spike_array_add_single)(unsigned int, float) = &SpikeArray::add;

static void spike_array_add_numpy(SpikeArray &self, const np::ndarray &indices, const np::ndarray &times) {
    check_indices_times_np_arrays(indices, times);

    const auto *indices_start = (unsigned int *) indices.get_data();
    const auto *indices_end = indices_start + indices.shape(0);
    const auto *times_start = (float *) times.get_data();

    self.add(indices_start, indices_end, times_start);
}

/*
 * SpikingNetwork wrappers
 */

static std::shared_ptr<FCLayer> (SpikingNetwork::*spiking_network_add_fc_layer)
        (const FCLayerDescriptor &) = &SpikingNetwork::add_layer;

/*
 * EvSpikeSim module definition
 */

BOOST_PYTHON_MODULE (evspikesim) {
    Py_Initialize();
    np::initialize();

    class_<Spike>("Spike", init<unsigned int, float>())
            .def_readwrite("index", &Spike::index)
            .def_readwrite("time", &Spike::time)
            .def(self == self)
            .def(self != self)
            .def(self < self)
            .def(self <= self)
            .def(self > self)
            .def(self >= self)
            .def(self_ns::str(self_ns::self));

    class_<SpikeArray>("SpikeArray", no_init)
            .def("__init__", make_constructor(&spike_array_init, default_call_policies(),
                                              (arg("indices") = object(),
                                                      arg("times") = object())))
            .def("add", spike_array_add_single)
            .def("add", &spike_array_add_numpy)
            .def("sort", &SpikeArray::sort)
            .def("clear", &SpikeArray::clear)
            .def("empty", &SpikeArray::empty)
            .def("__len__", &SpikeArray::n_spikes)
            .def("__iter__", range(&SpikeArray::begin, &SpikeArray::end))
            .def(self_ns::str(self_ns::self))
            .def(self == self)
            .def(self != self)
            .def_readonly("n_spikes", &SpikeArray::n_spikes);

    class_<SpikingNetwork>("SpikingNetwork")
            .def("add_layer", spiking_network_add_fc_layer)
            .def("infer", make_function(&SpikingNetwork::infer, return_value_policy<reference_existing_object>()))
            .def("__len__", &SpikingNetwork::get_n_layers)
            .def("__iter__", range(&SpikingNetwork::begin, &SpikingNetwork::end))
            .def("__getitem__", &SpikingNetwork::operator[]<unsigned int>)
            .add_property("output_layer", &SpikingNetwork::get_output_layer);

    export_layers_module();
}