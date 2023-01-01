#include "python_interface/py_random.h"
#include "random.h"

PyObject *py_random_set_seed(PyObject *self, PyObject *args) {
    unsigned int seed;
    
    if(!PyArg_ParseTuple(args, "I", &seed))
        return 0;
    random_set_seed(seed);

    Py_INCREF(Py_None);
    return Py_None;
}
