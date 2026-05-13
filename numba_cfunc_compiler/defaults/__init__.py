from numba_cfunc_compiler.numba_config import NumbaTypeRegistry

is_supported_type = NumbaTypeRegistry.is_supported_type


def register_types():
    """Register all default type support (primitives, datetime, list, dict, struct)."""
    from numba_cfunc_compiler.defaults import (
        datetime_support,
        dict_support,
        list_support,
        primitive_support,
        struct_support,
        timedelta_support,
    )

    primitive_support.register()
    datetime_support.register()
    timedelta_support.register()
    list_support.register()
    dict_support.register()
    struct_support.register()


def register_all():
    """Register all default type support and source categories."""
    from numba_cfunc_compiler.source_registry import register_default_categories

    register_types()
    register_default_categories()
