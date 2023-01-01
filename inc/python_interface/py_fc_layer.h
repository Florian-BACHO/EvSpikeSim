#ifndef PY_FC_LAYER
#define PY_FC_LAYER
 
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <Python.h>
#include "fc_layer.h"

typedef struct {
    PyObject_HEAD;
    fc_layer_t *layer;
} py_fc_layer_t;

// Getters
PyObject *py_fc_layer_get_weights(py_fc_layer_t *self, void *closure);
PyObject *py_fc_layer_get_spike_counts(py_fc_layer_t *self, void *closure);
PyObject *py_fc_layer_get_spikes(py_fc_layer_t *self, void *closure);

// Setters
int py_fc_layer_set_weights(py_fc_layer_t *self, PyObject *value, void *closure);

// Mapping methods
Py_ssize_t py_fc_layer_len(py_fc_layer_t *self); // Returns number of neurons

PyMethodDef py_fc_layer_methods[1]; // Methods definitions
PyGetSetDef py_fc_layer_getset[4]; // Getters and setters definitions
PyMappingMethods py_fc_layer_mapping; // Methods definition for __len__
PyTypeObject py_fc_layer_type; // Type definition

#endif
