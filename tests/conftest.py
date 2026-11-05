"""Conftest: mock missing optional deps before any test module imports; shared fixtures."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# pytorch_wavelets requires pkg_resources (setuptools), which may be absent
# in some Python 3.13 environments. Only stub out the dependency if the import
# actually fails.
_MISSING_DEPS = [
    "pkg_resources",
    "pytorch_wavelets",
    "pytorch_wavelets.dtcwt",
    "pytorch_wavelets.dtcwt.transform2d",
    "pytorch_wavelets.dtcwt.coeffs",
    "pytorch_wavelets.dwt",
    "pytorch_wavelets.dwt.lowlevel",
]
for _mod in _MISSING_DEPS:
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except (ImportError, ModuleNotFoundError):
            sys.modules[_mod] = MagicMock()


@pytest.fixture(scope="session")
def _mlflow_store_dir(tmp_path_factory: pytest.TempPathFactory):
    return tmp_path_factory.mktemp("mlflow_store")


@pytest.fixture
def mlflow_store(_mlflow_store_dir, monkeypatch: pytest.MonkeyPatch) -> str:
    """A real local MLflow tracking store, selected through ``MLFLOW_TRACKING_URI``.

    One sqlite store per session (creating one takes ~10 s; mlflow 3.14 refuses the file
    store), so give every test its own experiment name. The test runs from the store's
    directory because a sqlite store writes artifacts under ``./mlruns``.
    """
    from mlflow.tracking._tracking_service import utils

    uri = f"sqlite:///{_mlflow_store_dir / 'mlflow.db'}"
    monkeypatch.setattr(utils, "_tracking_uri", None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(_mlflow_store_dir)
    return uri
