from llvmlite import ir
from numba.core import types


# These are functions rather than constants because ir types are mutable
# and should be created fresh for each use in different modules.
def i8():
    return ir.IntType(8)


def i32():
    return ir.IntType(32)


def i64():
    return ir.IntType(64)


def i8ptr():
    return ir.IntType(8).as_pointer()


def f64():
    return ir.DoubleType()


def get_or_declare_function(module, name, fnty):
    """Get existing function or declare it in the module."""
    if name in module.globals:
        return module.globals[name]
    return ir.Function(module, fnty, name=name)


def convert_to_i64(builder, value):
    """Convert an integer value to i64, sign-extending if needed."""
    target = i64()
    if value.type.width < 64:
        return builder.sext(value, target)
    return value


def convert_to_i8(builder, value):
    """Convert an integer value to i8, truncating or zero-extending as needed."""
    target = i8()
    if value.type.width > 8:
        return builder.trunc(value, target)
    elif value.type.width < 8:
        return builder.zext(value, target)
    return value


def get_llvm_type_for_numba_dtype(dtype):
    if dtype == types.int64:
        return i64()
    elif dtype == types.float64:
        return f64()
    elif dtype == types.int8:
        return i8()
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")


def prepare_int_key_for_lookup(builder, key, key_type):
    i8_ptr = i8ptr()
    i64_type = i64()

    if key_type == types.int8:
        key_converted = convert_to_i8(builder, key)
        key_lltype = i8()
        hash_val = builder.sext(key_converted, i64_type)
    else:  # int64
        key_converted = convert_to_i64(builder, key)
        key_lltype = i64_type
        hash_val = key_converted

    key_ptr = builder.alloca(key_lltype)
    builder.store(key_converted, key_ptr)
    key_char_ptr = builder.bitcast(key_ptr, i8_ptr)

    return key_char_ptr, hash_val


def alloc_value_buffer(builder, value_type):
    i8_ptr = i8ptr()
    val_lltype = get_llvm_type_for_numba_dtype(value_type)
    val_ptr = builder.alloca(val_lltype)
    val_char_ptr = builder.bitcast(val_ptr, i8_ptr)
    return val_ptr, val_char_ptr


def store_value_to_buffer(builder, value, dtype):
    i8_ptr = i8ptr()
    lltype = get_llvm_type_for_numba_dtype(dtype)

    # Convert value to target type
    if dtype == types.int64:
        converted = convert_to_i64(builder, value)
    elif dtype == types.int8:
        converted = convert_to_i8(builder, value)
    else:
        converted = value  # float64 - no conversion needed

    val_ptr = builder.alloca(lltype)
    builder.store(converted, val_ptr)
    val_char_ptr = builder.bitcast(val_ptr, i8_ptr)

    return val_ptr, val_char_ptr
