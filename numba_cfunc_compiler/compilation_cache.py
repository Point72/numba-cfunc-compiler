import logging
import os
import tempfile
import threading
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import llvmlite
import numba

import numba_cfunc_compiler

__all__ = [
    "CacheConfig",
    "CompilationCache",
    "compile_cached",
    "get_cache_config",
    "get_compilation_cache",
    "set_cache_config",
]

log = logging.getLogger("numba_cfunc_compiler")


def _cache_namespace() -> str:
    """Subdirectory per toolchain version, since each can change the IR for a key."""
    return f"ncc-{numba_cfunc_compiler.__version__}-numba-{numba.__version__}-llvmlite-{llvmlite.__version__}"


@dataclass(frozen=True)
class CacheConfig:
    """Process-wide cache settings.

    enabled:    reuse compiled IR within the process.
    directory:  also share it across processes; defaults to $NUMBA_CFUNC_COMPILER_CACHE_DIR.
    """

    enabled: bool = False
    directory: str | None = field(default_factory=lambda: os.environ.get("NUMBA_CFUNC_COMPILER_CACHE_DIR") or None)


class CompilationCache:
    def __init__(self, directory: str | os.PathLike | None = None) -> None:
        self.directory = Path(directory) / _cache_namespace() if directory else None
        self._memory: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            llvm_ir = self._memory.get(key)
        if llvm_ir is not None or self.directory is None:
            return llvm_ir
        try:
            llvm_ir = self._path(key).read_text()
        except OSError:
            return None
        with self._lock:
            self._memory[key] = llvm_ir
        return llvm_ir

    def put(self, key: str, llvm_ir: str) -> None:
        with self._lock:
            self._memory[key] = llvm_ir
        if self.directory is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=self.directory, prefix=f".{key}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as file:
                    file.write(llvm_ir)
                os.replace(tmp_path, self._path(key))
            except BaseException:
                Path(tmp_path).unlink(missing_ok=True)
                raise
        except OSError as e:
            log.warning(f"Failed to write compilation cache entry {key} to {self.directory}: {e}")

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.ll"


_config = CacheConfig()
_cache: CompilationCache | None = None


def set_cache_config(config: CacheConfig) -> None:
    """Replace the cache settings and start an empty cache."""
    global _config, _cache
    _config = config
    _cache = CompilationCache(config.directory) if config.enabled else None


def get_cache_config() -> CacheConfig:
    return _config


def get_compilation_cache() -> CompilationCache | None:
    """Return the process-wide cache, or None when disabled."""
    return _cache


def compile_cached(
    name: str,
    semantic_key: str,
    unkeyable_globals: Sequence[str],
    compile_fn: Callable[[], tuple[Any, str]],
) -> tuple[Any | None, str]:
    """Return (compiled_func, llvm_ir), calling compile_fn only on a miss.

    compiled_func is None on a hit. Functions with unkeyable globals are never cached.
    """
    cache = get_compilation_cache()
    if cache is not None and unkeyable_globals:
        warnings.warn(
            f"Not caching '{name}': Numba compiles in the value of global(s) {', '.join(unkeyable_globals)}, "
            "which are not plain constants the cache key can cover",
            stacklevel=3,  # the caller of create_compiled_func
        )
        cache = None
    if cache is not None and (llvm_ir := cache.get(semantic_key)) is not None:
        return None, llvm_ir

    compiled_func, llvm_ir = compile_fn()
    # IR with this process's object addresses is only valid here.
    if cache is not None and not compiled_func._library.has_dynamic_globals:
        cache.put(semantic_key, llvm_ir)
    return compiled_func, llvm_ir
