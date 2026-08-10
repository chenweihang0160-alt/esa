"""抓取三个计算器的**线上真实返回值**，落成缓存。

为什么要有这个脚本
------------------
`tools_exec.py` 里的三个计算器原本是我自己重写的一版，返回 `{"value": 44}`。
而后端真实返回的是 `{"expression": "20+4*6", "result": 44}` ——
字段名完全不同，`math_solver` 更是根本没有 `latex` 字段（数据里 22 条回答正是引用它写的）。
模型学会消费一个线上不存在的结构，等于白训。

所以改成和 `arxiv_real.json` 同一套办法：**跑真实实现，把真实输出存下来**，
生成器只查表不计算。缓存本身就是「这些观测值不是编的」的证据，也保证可复现
（赛题《03—技术方案说明》明确要求可复现）。

复刻还是执行？—— 执行。复刻一份代码到本地，下次后端改了实现我们不会知道；
执行真实函数，改了就是改了，重跑一次立刻暴露。

用法
----
    python3 dataset/tools/capture_math_outputs.py                 # 自动下载后端快照
    python3 dataset/tools/capture_math_outputs.py --repo ~/esa    # 用本地已有的副本

⚠️ 只读后端仓库，不改它的任何文件。下载的副本落在临时目录，不入库。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CALC_SEEDS = ROOT / "dataset/seeds/calculators.yaml"
ERROR_SEEDS = ROOT / "dataset/seeds/tool_errors.yaml"
OUT = ROOT / "dataset/data/cache/math_real.json"

TARBALL = "https://codeload.github.com/LoveLearnLearning/esa/tar.gz/refs/heads/main"
MATH_SOURCES = [
    "backend/agent/tools/math_tools/calculator.py",
    "backend/agent/tools/math_tools/bitwise_calculator.py",
    "backend/agent/tools/math_tools/math_solver.py",
    "backend/agent/tools/math_tools/_base_evaluator.py",
]
TOOLS = ("calculator", "bitwise_calculator", "math_solver")


def cache_key(tool: str, args: dict) -> str:
    """缓存的键。参数排序后序列化，保证同一组参数永远得到同一个键。"""
    return f"{tool}|{json.dumps(args, ensure_ascii=False, sort_keys=True)}"


def fetch_repo(dest: Path) -> Path:
    """下载后端快照并解包，返回仓库根目录。"""
    print(f"下载后端快照 → {dest}")
    tgz = dest / "esa.tar.gz"
    urllib.request.urlretrieve(TARBALL, tgz)
    with tarfile.open(tgz) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            # filter= 是 PEP 706 加的，老一点的 3.9/3.10/3.11 补丁版没有。
            # 解到临时目录、解完就删，退回旧行为可以接受。
            tf.extractall(dest)  # noqa: S202
    roots = [p for p in dest.iterdir() if p.is_dir() and (p / "backend").exists()]
    if not roots:
        raise SystemExit("解包后找不到 backend/ 目录，仓库结构可能变了")
    return roots[0]


def source_fingerprint(repo: Path) -> dict[str, str]:
    """三个计算器源文件的 sha256。

    这是最贴切的溯源信息：它钉住的正是产出这些观测值的那段代码。
    指纹一变就说明后端改了实现，缓存必须重抓。
    """
    out = {}
    for rel in MATH_SOURCES:
        p = repo / rel
        if not p.exists():
            raise SystemExit(f"后端仓库里找不到 {rel}，路径可能变了")
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def load_backend_tools(repo: Path):
    """import 真实实现。失败要吵，不要退回本地近似。"""
    sys.path.insert(0, str(repo))
    try:
        mods = {t: importlib.import_module(f"backend.agent.tools.math_tools.{t}") for t in TOOLS}
    except ImportError as exc:
        raise SystemExit(
            f"无法 import 后端 math_tools：{exc}\n"
            "这些模块依赖 requests/sympy，先 pip install 再重试。"
            "注意不要改用本地实现顶替 —— 那正是这次要修掉的问题。"
        ) from exc
    return {t: getattr(mods[t], t) for t in TOOLS}


def collect_jobs() -> list[tuple[str, dict]]:
    """从种子库里收集所有需要真实执行的调用。"""
    jobs: list[tuple[str, dict]] = []
    seen: set[str] = set()

    def add(tool: str, args: dict) -> None:
        k = cache_key(tool, args)
        if k not in seen:
            seen.add(k)
            jobs.append((tool, args))

    calc = yaml.safe_load(CALC_SEEDS.read_text(encoding="utf-8"))
    for seed in calc["calculator"]:
        add("calculator", {"expression": seed["expr"]})
    for seed in calc["bitwise_calculator"]:
        add("bitwise_calculator", {"expression": seed["expr"]})
    for seed in calc["math_solver"]:
        args = {"operation": seed["op"], "expression": seed["expr"], "variable": seed["var"]}
        args.update(seed.get("extra") or {})
        add("math_solver", args)

    # tool_error 种子库里的失败调用（以及「改参数重试」那一次修正后的调用）
    if ERROR_SEEDS.exists():
        errs = yaml.safe_load(ERROR_SEEDS.read_text(encoding="utf-8")) or {}
        for item in errs.get("calculators", []):
            add(item["tool"], item["args"])
            if item.get("retry_args") is not None:
                add(item["tool"], item["retry_args"])
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取三个计算器的线上真实返回值")
    ap.add_argument("--repo", help="本地后端仓库副本；不给就自动下载快照")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    tmp = None
    if args.repo:
        repo = Path(args.repo).expanduser().resolve()
        if not (repo / "backend").exists():
            raise SystemExit(f"{repo} 下没有 backend/ 目录")
    else:
        tmp = Path(tempfile.mkdtemp(prefix="esa_backend_"))
        repo = fetch_repo(tmp)

    try:
        fingerprint = source_fingerprint(repo)
        fns = load_backend_tools(repo)
        jobs = collect_jobs()

        entries: dict[str, dict] = {}
        n_err = 0
        for tool, call_args in jobs:
            # 这三个工具**不抛异常**，失败也是正常返回一个带 error 键的 dict
            # （calculator.py:145-150 / bitwise_calculator.py:157-162 / math_solver.py:298-304）。
            # 这一点本身就是要教给模型的：同一个 Agent 里存在两套失败表示。
            result = fns[tool](**call_args)
            entries[cache_key(tool, call_args)] = result
            if isinstance(result, dict) and "error" in result:
                n_err += 1

        payload = {
            "_meta": {
                "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": "github.com/LoveLearnLearning/esa",
                "source_sha256": fingerprint,
                "note": "由 dataset/tools/capture_math_outputs.py 执行后端真实函数产出，禁止手改",
            },
            "entries": entries,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"\n抓到 {len(entries)} 条真实返回（其中 {n_err} 条是失败返回）→ {out_path.relative_to(ROOT)}")
        for rel, h in fingerprint.items():
            print(f"  {h}  {rel}")
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
