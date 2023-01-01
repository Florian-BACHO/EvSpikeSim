#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL SIMULATOR_ARRAY_API

#include "python_interface/py_fc_layer.h"
#include "random.h"
#include "numpy/arrayobject.h"

PyMethodDef py_fc_layer_methods[1] = {
				      {NULL}  /* Sentinel */
};

PyGetSetDef py_fc_layer_getset[4] = {
				     {"weights", (getter) py_fc_layer_get_weights,
				      (setter) py_fc_layer_set_weights, "Layer weights", NULL},
				     {"spike_counts", (getter) py_fc_layer_get_spike_counts,
				      NULL, "Layer spike counts", NULL},
				     {"spikes", (getter) py_fc_layer_get_spikes,
				      NULL, "Layer spike events", NULL},
				     {NULL}  /* Sentinel */
};

PyMappingMethods py_fc_layer_mapping = {
					(lenfunc) py_fc_layer_len,
					NULL,
					NULL
};

PyTypeObject py_fc_layer_type = {
				 PyVarObject_HEAD_INIT(NULL, 0)
				 .tp_name = "simulator.FCLayer",
				 .tp_doc = "Fully-connected layer wrapper object",
				 .tp_basicsize = sizeof(py_fc_layer_t),
				 .tp_itemsize = 0,
				 .tp_flags = Py_TPFLAGS_DEFAULT,
				 .tp_new = PyType_GenericNew,
				 .tp_methods = py_fc_layer_methods,
				 .tp_as_mapping = &py_fc_layer_mapping,
				 .tp_getset = py_fc_layer_getset,
};

PyObject *py_fc_layer_get_weights(py_fc_layer_t *self, void *closure) {
    unsigned int n_neurons = self->layer->params.n_neurons;
    unsigned int n_inputs = self->layer->params.n_inputs;
    npy_intp dims[] = {(npy_intp)n_neurons, (npy_intp)n_inputs};

    return PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, self->layer->weights);
}

int py_fc_layer_set_weights(py_fc_layer_t *self, PyObject *value, void *closure) {
    PyArrayObject *weights_obj;
    const float *weights;

    weights_obj = PyArray_GETCONTIGUOUS((PyArrayObject *)value); // Unsure c-contiguous array for memcpy       
    weights = (float *)PyArray_DATA(weights_obj);

    fc_layer_set_weights(self->layer, weights);

    Py_INCREF(Py_None);
    return 0;
}

PyObject *py_fc_layer_get_spike_counts(py_fc_layer_t *self, void *closure) {
    npy_intp n_neurons = (npy_intp)self->layer->params.n_neurons;

    return PyArray_SimpleNewFromData(1, &n_neurons, NPY_INT, self->layer->n_spikes);
}

PyObject *py_fc_layer_get_spikes(py_fc_layer_t *self, void *closure) {
    const spike_list_t *spikes = self->layer->post_spikes;
    npy_intp n_spikes = (npy_intp)self->layer->total_n_spikes;
    unsigned int *spike_idx_ptr;
    float *spike_time_ptr;
    PyArrayObject *spike_indices = (PyArrayObject *)PyArray_SimpleNew(1, &n_spikes, NPY_INT);
    PyArrayObject *spike_times = (PyArrayObject *)PyArray_SimpleNew(1, &n_spikes, NPY_FLOAT);

    if (spike_indices == 0 || spike_times == 0)
        return 0;
    for (unsigned int i = 0; i < n_spikes; i++) {
        spike_idx_ptr = (unsigned int *)PyArray_GETPTR1(spike_indices, i);
        spike_time_ptr = (float *)PyArray_GETPTR1(spike_times, i);
        *spike_idx_ptr = spikes->index;
        *spike_time_ptr = spikes->time;
        spikes = spikes->next;
    }
    return PyTuple_Pack(2, spike_indices, spike_times);
}

Py_ssize_t py_fc_layer_len(py_fc_layer_t *self) {
    return (Py_ssize_t)self->layer->params.n_neurons;
}
