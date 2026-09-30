"""``python -m datasetfactory`` 入口。

与项目根的 ``main.py`` 等价，区别是走正常的包导入路径：``main.py``
自己把 ``src/`` 塞进 ``sys.path`` 以便「零安装」直接跑；而 ``-m`` 形式
要求包本身可导入，即已经 ``pip install -e .``，或临时
``PYTHONPATH=src python -m datasetfactory ...``。

安装之后（``pip install -e .``）也可以直接用 ``datasetfactory``
命令，那由 pyproject 的 ``[project.scripts]`` 提供。
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
