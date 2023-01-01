#define PY_ARRAY_UNIQUE_SYMBOL SIMULATOR_ARRAY_API

#include "python_interface/py_module_init.h"
#include "python_interface/py_network.h"
#include "python_interface/py_fc_layer.h"
#include "python_interface/py_random.h"
#include "random.h"
#include "numpy/arrayobject.h"

PyMethodDef py_module_methods[2] = {
				    {"set_seed", (PyCFunction) py_random_set_seed, METH_VARARGS,
				     "Sets seed for random generator."
				    },
				    {NULL}  /* Sentinel */
};

PyModuleDef py_module = {
    PyModuleDef_HEAD_INIT,
    "EvSpikeSim",
    "An Event-Based Spiking Neural Network Simulator",
    -1,
    py_module_methods
};

PyMODINIT_FUNC PyInit_evspikesim(void) {
    PyObject* module;

    import_array(); // Numpy

    if (PyType_Ready(&py_network_type) < 0 || PyType_Ready(&py_fc_layer_type) < 0)
        return 0;

    module = PyModule_Create(&py_module);
    if (module == NULL)
        return NULL;

    Py_INCREF(&py_network_type);
    if (PyModule_AddObject(module, "Network", (PyObject *) &py_network_type) < 0) {
        Py_DECREF(&py_network_type);
        Py_DECREF(module);
        return NULL;
    }
    
    Py_INCREF(&py_fc_layer_type);
    if (PyModule_AddObject(module, "FCLayer", (PyObject *) &py_fc_layer_type) < 0) {
        Py_DECREF(&py_fc_layer_type);
        Py_DECREF(module);
        return NULL;
    }

    random_set_seed(time(0));
    
    return module;
}
