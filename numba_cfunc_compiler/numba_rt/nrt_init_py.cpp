// Minimal Python module using raw C API (no Boost.Python needed)
// This module exists only so we can import it and get __file__ for llvm.load_library_permanently()

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyModuleDef py_nrt_init_module = {
    PyModuleDef_HEAD_INIT,
    "_py_nrt_init",  // module name
    NULL,            // module documentation
    -1,              // size of per-interpreter state (-1 = global state)
    NULL,            // methods
    NULL,            // slots
    NULL,            // traverse
    NULL,            // clear
    NULL             // free
};

PyMODINIT_FUNC PyInit__py_nrt_init(void) {
    return PyModule_Create(&py_nrt_init_module);
}
