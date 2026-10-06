import shutil
import tarfile
from pathlib import Path

from hatchling.builders.sdist import SdistBuilder


def test_sdist_excludes_compiled_extensions(tmp_path):
    root = Path(__file__).resolve().parents[2]
    project = tmp_path / "project"
    shutil.copytree(root / "numba_cfunc_compiler", project / "numba_cfunc_compiler")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(root / name, project / name)

    extensions = (".so", ".dll", ".dylib", ".pyd")
    for suffix in extensions:
        (project / "numba_cfunc_compiler" / "numba_rt" / f"_py_nrt_init{suffix}").write_bytes(b"prebuilt binary")

    output = tmp_path / "dist"
    output.mkdir()
    filename = next(SdistBuilder(str(project)).build(directory=str(output)))
    with tarfile.open(output / filename) as archive:
        names = archive.getnames()

    assert not any(name.endswith(extensions) for name in names)
    assert any(name.endswith("/numba_rt/nrt_init_py.cpp") for name in names)
    assert any(name.endswith("/numba_rt/cext/dictobject.c") for name in names)
    assert any(name.endswith("/numba_rt/cext/listobject.c") for name in names)
