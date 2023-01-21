#ifndef PY_NETWORK
#define PY_NETWORK

#include <Python.h>
#include "network.h"

typedef struct {
    PyObject_HEAD;
    network_t *network;
} py_network_t;

// Constructor and destructor
PyObject *py_network_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
int py_network_init(py_network_t *self, PyObject *args, PyObject *kwds);
void py_network_dealloc(py_network_t *self);

// Methods
PyObject *py_network_add_fc_layer(py_network_t *self, PyObject *args);
PyObject *py_network_reset(py_network_t *self, PyObject *Py_UNUSED(ignored));
PyObject *py_network_infer(py_network_t *self, PyObject *args);

// Getters

PyObject *py_network_get_output_layer(py_network_t *self, void *closure);

// Mapping methods
Py_ssize_t py_network_len(py_network_t *self); // Returns number of layers
PyObject *py_network_get_item(py_network_t *self, PyObject *key);

PyMethodDef py_network_methods[4]; // Methods definitions
PyGetSetDef py_network_getset[2]; // Getters and setters definitions
PyMappingMethods py_network_mapping; // Methods definition for __len__ and __getitem__
PyTypeObject py_network_type; // Type definition

#endif
