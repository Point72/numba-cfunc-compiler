/**
 * Common definitions for Numba's typed list and dict C extensions.
 */
#ifndef NUMBA_CEXT_H_
#define NUMBA_CEXT_H_

#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

/* Symbol visibility macros */
#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if defined(_MSC_VER)
#define VISIBILITY_HIDDEN
#define VISIBILITY_GLOBAL __declspec(dllexport)
#elif (__has_attribute(visibility) || (defined(__GNUC__) && __GNUC__ >= 4))
#define VISIBILITY_HIDDEN __attribute__((visibility("default")))
#define VISIBILITY_GLOBAL __attribute__((visibility("default")))
#else
#define VISIBILITY_HIDDEN
#define VISIBILITY_GLOBAL
#endif

/* Export macros for functions and data */
#define NUMBA_EXPORT_FUNC(_rettype) VISIBILITY_HIDDEN _rettype
#define NUMBA_EXPORT_DATA(_vartype) VISIBILITY_HIDDEN _vartype
#define NUMBA_GLOBAL_FUNC(_rettype) VISIBILITY_GLOBAL _rettype

/* Python-compatible types (standalone, no Python.h dependency) */
typedef struct _object PyObject;
typedef long Py_ssize_t;
typedef unsigned long Py_hash_t;

#define PY_SSIZE_T_MAX ((Py_ssize_t)(((size_t)-1) >> 1))
#define PY_SSIZE_T_MIN (-PY_SSIZE_T_MAX - 1)

#ifndef assert
#define assert(expr)                                                                                                   \
  ((expr) ? (void)0 : (fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", #expr, __FILE__, __LINE__), abort()))
#endif

/* Utility functions */
static inline Py_ssize_t aligned_size(Py_ssize_t sz) {
  Py_ssize_t alignment = sizeof(void*);
  return sz + (alignment - sz % alignment) % alignment;
}

static inline int mem_cmp_zeros(const void* p, size_t size) {
  const unsigned char* it = (const unsigned char*)p;
  size_t i;
  int diff = 0;
  for(i = 0; i < size; ++i) {
    if(it[i] != 0)
      diff += 1;
  }
  return diff;
}

static inline void* aligned_pointer(void* p) { return (void*)aligned_size((Py_ssize_t)(size_t)p); }

/* Include typed container headers */
#include "dictobject.h"
#include "listobject.h"

#endif /* NUMBA_CEXT_H_ */
