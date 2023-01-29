//
// Created by Florian Bacho on 28/01/23.
//

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "MakeSubmodule.h"
#include "LayersModule.h"
#include "Layers/LayerDescriptor.h"
#include "Layers/FCLayerDescriptor.h"
#include "Layers/Layer.h"
#include "Layers/FCLayer.h"

using namespace boost::python;
using namespace EvSpikeSim;
namespace np = boost::python::numpy;

/*
 * Layer wrappers
 */

static tuple get_weights_dim_tuple(const std::vector<unsigned int> &dims) {
    list tmp;

    for (auto dim : dims)
        tmp.append(dim);
    return tuple(tmp);
}

static tuple get_weights_stride_tuple(const std::vector<unsigned int> &dims) {
    list tmp;
    unsigned int current_offset = sizeof(float);

    for (auto it = dims.end() - 1; it != dims.begin() - 1; it--) {
        tmp.append(current_offset);
        current_offset *= *it;
    }
    tmp.reverse();
    return tuple(tmp);
}

static np::ndarray layer_get_np_weights(Layer &layer) {
    np::dtype dt = np::dtype::get_builtin<float>();
    const auto &dims = layer.get_weights().get_dims();
    auto stride = get_weights_stride_tuple(dims);
    auto shape = get_weights_dim_tuple(dims);

    return np::from_data(layer.get_weights().c_ptr(), dt, shape, stride, object());
}

static void layer_set_np_weights(Layer &layer, const np::ndarray &new_weights) {
    auto &weights = layer.get_weights();
    auto size = weights.size();
    const auto &dims = weights.get_dims();
    const auto *data = (const float *) new_weights.get_data();

    if (new_weights.get_dtype() != np::dtype::get_builtin<float>()) {
        PyErr_SetString(PyExc_TypeError, "New weights have incorrect data type");
        throw_error_already_set();
    }
    if (new_weights.get_nd() != weights.get_n_dims()) {
        PyErr_SetString(PyExc_TypeError, "Number of dimensions in new weights does not match the number dimensions of "
                                         "weights in the layer");
        throw_error_already_set();
    }
    for (auto i = 0; i < new_weights.get_nd(); i++)
        if (new_weights.shape(i) != dims[i]) {
            PyErr_SetString(PyExc_TypeError, "Number of indices does not match the number of times");
            throw_error_already_set();
        }
    std::copy(data, data + size, weights.c_ptr());
}

static np::ndarray layer_get_np_n_spikes(const Layer &layer) {
    np::dtype dt = np::dtype::get_builtin<unsigned int>();
    const std::vector<unsigned int> &n_spikes = layer.get_n_spikes();
    auto stride = make_tuple(sizeof(unsigned int));
    auto shape = make_tuple(n_spikes.size());

    return np::from_data(n_spikes.data(), dt, shape, stride, object());
}

void export_layers_module() {
    MAKE_SUBMODULE(evspikesim, layers)

    np::initialize();

    class_<LayerDescriptor>("LayerDescriptor", init<unsigned int, unsigned int, float, float>())
            .def_readonly("n_inputs", &FCLayerDescriptor::n_inputs)
            .def_readonly("n_neurons", &FCLayerDescriptor::n_neurons)
            .def_readonly("tau_s", &FCLayerDescriptor::tau_s)
            .def_readonly("tau", &FCLayerDescriptor::tau)
            .def_readonly("threshold", &FCLayerDescriptor::threshold);

    class_<FCLayerDescriptor, bases<LayerDescriptor>>("FCLayerDescriptor",
                                                      init<unsigned int, unsigned int, float, float>());

    class_<Layer, boost::noncopyable>("Layer", no_init)
            .add_property("descriptor", make_function(&Layer::get_descriptor,
                                                      return_value_policy<reference_existing_object>()))
            .add_property("weights", &layer_get_np_weights, &layer_set_np_weights)
            .add_property("post_spikes", make_function(&Layer::get_post_spikes,
                                                       return_value_policy<reference_existing_object>()))
            .add_property("n_spikes", layer_get_np_n_spikes);
    register_ptr_to_python<std::shared_ptr<Layer>>();

    class_<FCLayer, bases<Layer>, boost::noncopyable>("FCLayer", no_init);
    register_ptr_to_python<std::shared_ptr<FCLayer>>();
}