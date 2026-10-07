"""Host type adapters and registrations for built-in value families."""


def register_types():
    """Register all built-in host type adapters."""
    from numba_cfunc_compiler.types.builtin import datetime, primitive, timedelta
    from numba_cfunc_compiler.types.builtin.array import host as array_host
    from numba_cfunc_compiler.types.builtin.dict import host as dict_host
    from numba_cfunc_compiler.types.builtin.list import host as list_host

    primitive.register()
    datetime.register()
    timedelta.register()
    array_host.register()
    list_host.register()
    dict_host.register()
