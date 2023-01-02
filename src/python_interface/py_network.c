#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SIMULATOR_ARRAY_API

#include "python_interface/py_network.h"
#include "python_interface/py_fc_layer.h"
#include "random.h"
#include "numpy/arrayobject.h"

PyMethodDef py_network_methods[4] = {
				     {"add_fc_layer", (PyCFunction) py_network_add_fc_layer, METH_VARARGS,
				      "Adds a fully-connected layer to the network."},
				     {"reset", (PyCFunction) py_network_reset, METH_NOARGS,
				      "Resets the state of the network."},
				     {"infer", (PyCFunction) py_network_infer, METH_VARARGS,
				      "Infers the network with the given input spikes."
				     },
				     {NULL}  /* Sentinel */
};

PyGetSetDef py_network_getset[2] = {
				    {"output_layer", (getter) py_network_get_output_layer,
				     NULL, "Output layer", NULL},
				    {NULL}  /* Sentinel */
};

PyMappingMethods py_network_mapping = {
				       (lenfunc) py_network_len,
				       (binaryfunc) py_network_get_item,
				       NULL
};

PyTypeObject py_network_type = {
				PyVarObject_HEAD_INIT(NULL, 0)
				.tp_name = "simulator.Network",
				.tp_doc = "Network wrapper object",
				.tp_basicsize = sizeof(py_network_t),
				.tp_itemsize = 0,
				.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
				.tp_new = py_network_new,
				.tp_dealloc = (destructor) py_network_dealloc,
				.tp_methods = py_network_methods,
				.tp_as_mapping = &py_network_mapping,
				.tp_getset = py_network_getset,
};

void py_network_dealloc(py_network_t *self) {
    network_destroy(&self->network);
    
    Py_TYPE(self)->tp_free((PyObject *) self);
}

PyObject *py_network_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    py_network_t *self;
    self = (py_network_t *) type->tp_alloc(type, 0);
    if (self != NULL) {
	self->network = network_init();
    }
    return (PyObject *) self;
}

static inline float init_fct(void) {
    return random_uniform_float(-1.0f, 1.0f);
}

PyObject *py_network_add_fc_layer(py_network_t *self, PyObject *args) {
    fc_layer_params_t params;
    unsigned int n_inputs, n_neurons;
    float tau_s, threshold;

    // Parse arguments
    if(!PyArg_ParseTuple(args, "IIff", &n_inputs, &n_neurons, &tau_s, &threshold))
        return 0;

    params = fc_layer_params_new(n_inputs, n_neurons, tau_s, threshold);
    if (network_add_fc_layer(&self->network, params, &init_fct) == 0)
	return 0;

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *py_network_reset(py_network_t *self, PyObject *Py_UNUSED(ignored)) {
    network_reset(&self->network);
    Py_INCREF(Py_None);
    return Py_None;
}

// Convert arguments into a spike list
static spike_list_t *get_input_spikes(PyObject *args) {
    PyArrayObject *spike_indices, *spike_times;
    int n_spikes, spike_idx;
    float spike_time;
    spike_list_t *out = 0;

    if(!PyArg_ParseTuple(args, "OO", &spike_indices, &spike_times))
	return 0;
    n_spikes = PyArray_DIMS(spike_indices)[0];

    for (int i = 0; i < n_spikes; i++) {
	spike_idx = *(unsigned int *)PyArray_GETPTR1(spike_indices, i);
        spike_time = *(float *)PyArray_GETPTR1(spike_times, i);
        out = spike_list_add(out, spike_idx, spike_time);
        if (out == 0)
            return 0;
    }
    return out;
}

PyObject *py_network_infer(py_network_t *self, PyObject *args) {
    spike_list_t *input_spikes = get_input_spikes(args);
    const spike_list_t *output_spikes;

    if (input_spikes == 0)
        return 0;
    output_spikes = network_infer(&self->network, input_spikes);
    if (output_spikes == 0)
	return 0;
    spike_list_destroy(input_spikes);

    Py_INCREF(Py_None);
    return Py_None;
}

PyObject *py_network_get_output_layer(py_network_t *self, void *closure) {
    return py_network_get_item(self, PyLong_FromLong(self->network.n_layers - 1));
}

Py_ssize_t py_network_len(py_network_t *self) {
    return (Py_ssize_t)self->network.n_layers;
}

PyObject *py_network_get_item(py_network_t *self, PyObject *key) {
    unsigned int layer_idx;
    py_fc_layer_t *fc_out;
    PyObject *out;
    
    if (!PyLong_Check(key))
	return 0;
    layer_idx = (unsigned int) PyLong_AsLong(key);
    if (layer_idx >= self->network.n_layers)
	return 0;

    switch (self->network.layer_types[layer_idx]) {
    case FC:
	fc_out = (py_fc_layer_t *)PyType_GenericNew(&py_fc_layer_type, NULL, NULL);
	if (fc_out == 0)
	    return 0;
	fc_out->layer = (fc_layer_t *)self->network.layers[layer_idx];
	out = (PyObject *)fc_out;
	break;
    default:
	return 0;
    }
    //Py_DECREF(out);
    return out;
}
