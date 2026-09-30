"""命令行入口。

    python main.py run     --config configs/default.json --out out
    python main.py explain --config configs/default.json --sample-id wiki:ab12cd34ef56
    python main.py dag     --config configs/default.json
    python main.py list    --config configs/default.json
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .config import load_config
from .pipeline import run_pipeline
from .report import build_report, explain_sample, render_explanation, write_artifacts


def _run(args: argparse.Namespace):
    cfg = load_config(args.config)
    result = run_pipeline(cfg)

    if args.out:
        written = write_artifacts(result, args.out)
    else:
        written = {}

    if not args.quiet:
        print(build_report(result))
        if written:
            print("产物：")
            for name, path in written.items():
                print(f"  {name:14s} {path}")
    return 0


def _explain(args: argparse.Namespace):
    cfg = load_config(args.config)
    result = run_pipeline(cfg)
    expl = explain_sample(result, args.sample_id)

    if args.json:
        import json

        print(json.dumps(expl, ensure_ascii=False, indent=2))
    else:
        print(render_explanation(expl))
        # 只有在「完全查不到这个 id」时才给相近建议；
        # 被丢弃的样本是查到了的（status=rejected），不需要建议。
        if expl["status"] == "unknown":
            known = result.known_ids()
            needle = args.sample_id.split(":")[-1][:6]
            near = [i for i in known if needle and needle in i][:5]
            if near:
                print("\n你是不是想找：")
                for i in near:
                    print(f"  {i}")
            elif known:
                print(f"\n可用 sample_id 共 {len(known)} 个，例如：")
                for i in known[:5]:
                    print(f"  {i}")

    # 查到了（保留或被丢弃）都算成功；只有完全未知才返回非零
    return 0 if expl["status"] != "unknown" else 1


def _dag(args: argparse.Namespace):
    cfg = load_config(args.config)
    result = run_pipeline(cfg)
    if args.dot:
        print(result.dag.to_dot())
    else:
        print(result.dag.to_ascii())
        print()
        summary = result.dag.summary()
        print(f"nodes={summary['nodes']} edges={summary['edges']} by_kind={summary['by_kind']}")
    return 0


def _list(args: argparse.Namespace):
    cfg = load_config(args.config)
    result = run_pipeline(cfg)
    print(f"# 保留 {result.dataset.size()} 条")
    for name in sorted(result.dataset.splits):
        for s in result.dataset.splits[name]:
            print(f"  [kept/{name}] {s.sample_id}  chars={s.chars}  "
                  f"q={s.metric('quality.score'):.3f}  c={s.metric('contamination.ratio'):.3f}")
    print(f"# 丢弃 {len(result.rejections)} 条")
    for r in result.rejections:
        print(f"  [drop/{r.stage}] {r.sample_id}  rule={r.rule}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datasetfactory",
        description="DatasetFactory — 去重 / 质量评分 / 污染检测 / 数据谱系 DAG",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="执行完整流水线并落盘产物")
    p_run.add_argument("--config", default="configs/default.json")
    p_run.add_argument("--out", default="out", help="产物输出目录（传空串则只打印报告）")
    p_run.add_argument("--quiet", action="store_true", help="只落盘，不打印报告")
    p_run.set_defaults(func=_run)

    p_explain = sub.add_parser("explain", help="回答某个 sample 的三个问题")
    p_explain.add_argument("--config", default="configs/default.json")
    p_explain.add_argument("--sample-id", required=True)
    p_explain.add_argument("--json", action="store_true")
    p_explain.set_defaults(func=_explain)

    p_dag = sub.add_parser("dag", help="打印 Lineage DAG")
    p_dag.add_argument("--config", default="configs/default.json")
    p_dag.add_argument("--dot", action="store_true", help="输出 Graphviz DOT")
    p_dag.set_defaults(func=_dag)

    p_list = sub.add_parser("list", help="列出保留与丢弃的样本")
    p_list.add_argument("--config", default="configs/default.json")
    p_list.set_defaults(func=_list)

    return parser


def _force_utf8_stdio() -> None:
    """把 stdout / stderr 切到 UTF-8。

    报告用 ``✓`` / ``✗`` 标记过滤规则是否通过，这两个字符不在 GBK
    （cp936）码表内 —— 中文 Windows 的默认控制台编码下会直接抛
    ``UnicodeEncodeError`` 让 ``explain`` 崩溃。输出里的 ``→`` ``←``
    ``├─`` 等符号 GBK 里都有，所以只有这两行会炸。

    Python 3.6+ 在 Windows 控制台上按 UTF-8 字节流调用 ``WriteConsoleW``，
    因此切成 UTF-8 后中文与符号都能正常渲染；重定向到文件时也统一按
    UTF-8 落盘，与 ``out/`` 下其它产物保持一致。
    """
    for stream in (sys.stdout, sys.stderr):
        # pytest 的捕获对象和 StringIO 没有 reconfigure，跳过即可
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
