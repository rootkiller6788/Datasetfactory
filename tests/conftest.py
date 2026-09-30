"""pytest 公共夹具。

约定：测试从项目根目录运行，``src/`` 由本文件加入 ``sys.path``，
因此无需 ``pip install -e .`` 也能跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasetfactory.config import load_config  # noqa: E402
from datasetfactory.pipeline import run_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config(ROOT / "configs" / "default.json")


@pytest.fixture(scope="session")
def result(cfg):
    """跑一次完整流水线，整个测试会话复用。"""
    return run_pipeline(cfg)
