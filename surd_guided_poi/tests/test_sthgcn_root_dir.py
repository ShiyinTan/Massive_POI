"""get_root_dir must work when cwd is not under Spatio-Temporal-Hypergraph-Model."""

from __future__ import annotations

import os
from pathlib import Path

from surd_guided_poi.data.massive_steps import STHGCN_DIR
from surd_guided_poi.data.sthgcn_hypergraph import (
    _sthgcn_cwd,
    fixed_sthgcn_root,
)


def test_fixed_sthgcn_root_is_package_dir():
    assert Path(fixed_sthgcn_root()).resolve() == STHGCN_DIR.resolve()
    assert fixed_sthgcn_root().endswith("Spatio-Temporal-Hypergraph-Model")


def test_fixed_root_independent_of_cwd(tmp_path):
    prev = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert "Spatio-Temporal-Hypergraph-Model" not in os.getcwd().split(os.sep)
        assert Path(fixed_sthgcn_root()).is_dir()
        assert (Path(fixed_sthgcn_root()) / "preprocess").is_dir()
    finally:
        os.chdir(prev)


def test_sthgcn_cwd_contains_package_name(tmp_path):
    """Upstream get_root_dir does dirname_split.index('Spatio-Temporal-Hypergraph-Model')."""
    prev = os.getcwd()
    try:
        os.chdir(tmp_path)
        with _sthgcn_cwd():
            parts = os.getcwd().replace("\\", "/").split("/")
            assert "Spatio-Temporal-Hypergraph-Model" in parts
            assert Path(os.getcwd()).resolve() == STHGCN_DIR.resolve()
        # restored
        assert Path(os.getcwd()).resolve() == Path(tmp_path).resolve()
    finally:
        os.chdir(prev)


def test_upstream_get_root_dir_works_under_chdir():
    """Simulate utils.sys_util.get_root_dir while cwd is the package."""
    # Inline copy of upstream logic (avoid importing full STHGCN stack / PyG).
    def upstream_get_root_dir():
        dirname = os.getcwd()
        dirname_split = dirname.split("/")
        index = dirname_split.index("Spatio-Temporal-Hypergraph-Model")
        return "/".join(dirname_split[: index + 1])

    prev = os.getcwd()
    try:
        # Outside package → ValueError (the bug we fix).
        os.chdir("/tmp")
        try:
            upstream_get_root_dir()
            raised = False
        except ValueError:
            raised = True
        assert raised

        with _sthgcn_cwd():
            root = upstream_get_root_dir()
            assert root.endswith("Spatio-Temporal-Hypergraph-Model")
            assert Path(root).resolve() == STHGCN_DIR.resolve()
    finally:
        os.chdir(prev)
