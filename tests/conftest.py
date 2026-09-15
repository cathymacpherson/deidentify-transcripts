"""Shared test setup.

Every test runs in its own temporary working directory. Commands that write files relative to the
current directory (a report defaulting to ``label-report.csv``, for instance) would otherwise drop
them into the repository — and overwrite a real one belonging to whoever ran the tests.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    working = tmp_path / "cwd"
    working.mkdir()
    monkeypatch.chdir(working)
    yield working
    os.chdir(str(tmp_path.parent))
