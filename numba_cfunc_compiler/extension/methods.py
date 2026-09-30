"""Register payload methods with Numba on values and source wrappers."""

import inspect
from itertools import count

from numba.extending import intrinsic, overload_method, register_jitable

from numba_cfunc_compiler.types.native.input import InputValueType, unwrap_source
from numba_cfunc_compiler.types.native.state import CopyStateValueType

_REGISTERED_METHODS = {}
_FORWARDED_METHODS = {}
_FUNCTION_IDS = count()
_FROZEN_TYPES = set()


def method_registration_key(native_type, *, freeze=False):
    """Names and signatures available on a native payload type."""
    if freeze:
        _FROZEN_TYPES.add(native_type)
    return tuple(sorted((name, arg_names, kind) for (type_key, name), (_, arg_names, kind) in _REGISTERED_METHODS.items() if type_key == native_type))


def _check_registration(native_type, method_name, implementation, arg_names, kind):
    key = native_type, method_name
    if key in _REGISTERED_METHODS:
        registered, registered_args, registered_kind = _REGISTERED_METHODS[key]
        if registered is implementation and registered_args == arg_names and registered_kind == kind:
            return False
        raise ValueError(f"Method '{method_name}' is already registered on {native_type}")
    if native_type in _FROZEN_TYPES:
        raise RuntimeError(f"Methods for {native_type} are frozen after compilation")
    return True


@intrinsic
def payload_value(typingctx, value):
    if not isinstance(value, (InputValueType, CopyStateValueType)):
        return None
    sig = value.payload_type(value)

    def codegen(context, builder, signature, args):
        return context.make_helper(builder, value, value=args[0]).value

    return sig, codegen


def _build_overload(owner, native_type, method_name, arg_names, implementation, forward):
    function_id = next(_FUNCTION_IDS)
    overload_name = f"overload_value_method_{function_id}"
    implementation_name = f"implement_value_method_{function_id}"
    params = ", ".join(("value", *arg_names))
    arguments = ", ".join(f"unwrap_source({arg})" for arg in arg_names)
    if forward:
        call = f"payload_value(value).{method_name}({arguments})"
    else:
        call_args = f", {arguments}" if arguments else ""
        call = f"implementation(payload_value(value){call_args})" if owner is not type(native_type) else f"implementation(value{call_args})"
    if native_type is None:
        condition = "isinstance(value, owner)"
    elif owner is type(native_type):
        condition = "value == native_type"
    else:
        condition = "isinstance(value, owner) and value.payload_type == native_type"
    body = (
        f"def {overload_name}({params}):\n"
        f"    if {condition}:\n"
        f"        def {implementation_name}({params}):\n"
        f"            return {call}\n"
        f"        return {implementation_name}\n"
        "    return None\n"
    )
    namespace = {
        "owner": owner,
        "native_type": native_type,
        "implementation": implementation,
        "payload_value": payload_value,
        "unwrap_source": unwrap_source,
    }
    exec(body, namespace)  # noqa: S102 - fixed function template and validated identifiers
    overload_method(owner, method_name)(namespace[overload_name])


def register_value_method(native_type, method_name: str, implementation, arg_names: tuple[str, ...] | None = None):
    """Register one Numba implementation on a native value and its source wrappers.

    ``implementation`` receives the unwrapped native value first. It must be
    compilable by Numba and must have a fixed positional signature.
    """
    if not method_name.isidentifier():
        raise ValueError(f"Invalid method name {method_name!r}")
    if arg_names is None:
        parameters = list(inspect.signature(implementation).parameters.values())
        if not parameters or any(param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) for param in parameters):
            raise TypeError("Value methods require a fixed positional signature")
        arg_names = tuple(param.name for param in parameters[1:])
    if any(not arg.isidentifier() for arg in arg_names):
        raise ValueError("Method argument names must be identifiers")
    if not _check_registration(native_type, method_name, implementation, arg_names, "method"):
        return implementation
    compiled = register_jitable(inline="always")(implementation)
    _build_overload(type(native_type), native_type, method_name, arg_names, compiled, False)
    _build_overload(InputValueType, native_type, method_name, arg_names, compiled, False)
    _build_overload(CopyStateValueType, native_type, method_name, arg_names, compiled, False)
    _REGISTERED_METHODS[native_type, method_name] = (implementation, arg_names, "method")
    return implementation


def register_value_overload(native_type, method_name: str, factory):
    """Register an argument-sensitive Numba method factory on one value family."""
    if not method_name.isidentifier():
        raise ValueError(f"Invalid method name {method_name!r}")
    parameters = list(inspect.signature(factory).parameters.values())
    if not parameters or any(param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) for param in parameters):
        raise TypeError("Value method overloads require a fixed positional signature")
    arg_names = tuple(param.name for param in parameters[1:])
    if not _check_registration(native_type, method_name, factory, arg_names, "overload"):
        return factory
    forward_value_method(native_type, method_name, arg_names)
    function_id = next(_FUNCTION_IDS)
    params = ", ".join(("value", *arg_names))
    arguments = ", ".join(("value", *arg_names))
    namespace = {"native_type": native_type, "factory": factory}
    exec(  # noqa: S102 - fixed template and validated identifiers
        f"def overload_{function_id}({params}):\n    if value == native_type:\n        return factory({arguments})\n    return None\n",
        namespace,
    )
    overload_method(type(native_type), method_name)(namespace[f"overload_{function_id}"])
    _REGISTERED_METHODS[native_type, method_name] = (factory, arg_names, "overload")
    return factory


def forward_value_method(native_type, method_name: str, arg_names: tuple[str, ...] = ()):
    """Forward an existing native Numba method from input and copied state values."""
    if not method_name.isidentifier() or any(not name.isidentifier() for name in arg_names):
        raise ValueError("Invalid value method name or arguments")
    key = native_type, method_name
    if key in _REGISTERED_METHODS:
        return
    existing_args = _FORWARDED_METHODS.get(method_name)
    if existing_args is not None and existing_args != arg_names:
        raise ValueError(f"Method '{method_name}' has conflicting argument names")
    if existing_args is None:
        _build_overload(InputValueType, None, method_name, arg_names, None, True)
        _build_overload(CopyStateValueType, None, method_name, arg_names, None, True)
        _FORWARDED_METHODS[method_name] = arg_names
    _REGISTERED_METHODS[key] = (None, arg_names, "forward")
