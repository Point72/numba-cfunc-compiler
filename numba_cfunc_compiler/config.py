"""Process-wide native ABI configuration."""

from dataclasses import dataclass
from threading import Lock

__all__ = ["NumbaConfig", "get_numba_config", "set_numba_config"]


@dataclass(frozen=True)
class NumbaConfig:
    """Native enum widths used at the host application's FFI boundary."""

    enum_bit_width: int = 16
    enumset_bit_width: int = 128

    def __post_init__(self) -> None:
        if type(self.enum_bit_width) is not int or self.enum_bit_width not in (8, 16, 32, 64):
            raise ValueError("enum_bit_width must be one of 8, 16, 32, 64")
        if type(self.enumset_bit_width) is not int or self.enumset_bit_width <= 0 or self.enumset_bit_width % 8:
            raise ValueError("enumset_bit_width must be a positive multiple of 8")


_numba_config: NumbaConfig | None = None
_numba_config_lock = Lock()


def set_numba_config(*, enum_bit_width: int = 16, enumset_bit_width: int = 128) -> None:
    """Override the process-wide native ABI before its first use.

    The widths must match those of the host application. If overriding the
    defaults, call this during startup before using compiler or context APIs.
    """
    global _numba_config
    with _numba_config_lock:
        if _numba_config is not None:
            raise RuntimeError("Numba configuration is already set and cannot be changed")
        _numba_config = NumbaConfig(enum_bit_width, enumset_bit_width)


def get_numba_config() -> NumbaConfig:
    """Return the ABI configuration, selecting the defaults on first use."""
    global _numba_config
    if _numba_config is None:
        with _numba_config_lock:
            if _numba_config is None:
                _numba_config = NumbaConfig()
    return _numba_config
