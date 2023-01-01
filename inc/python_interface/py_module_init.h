#ifndef PY_MODULE_INIT
#define PY_MODULE_INIT

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <Python.h>

PyMODINIT_FUNC PyInit_evspikesim(void);

PyMethodDef py_module_methods[2]; // Methods definitions
PyModuleDef py_module; // Module definition

#endif
