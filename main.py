#!/usr/bin/env python3
"""DatasetFactory 零安装入口。

把 ``src/`` 加入 ``sys.path`` 后转交 ``datasetfactory.cli``，
因此无需 ``pip install`` 即可直接运行：

    python main.py run --config configs/default.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasetfactory.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
