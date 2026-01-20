from __future__ import annotations

"""
1) 输入多个 .v（分层设计），为每个 .v 构建 AST，并从 AST 抽取模块实例化关系图（谁实例化谁）
2) 从这些 AST 抽取每个模块的 I/O 接口表（模块名、端口名、方向 input/output、位宽）
3) 输入 target_signal（顶层信号），在跨模块数据依赖图上枚举 target_signal -> 叶子信号 的所有路径并打印

用法示例：
python3 example/verilog_hierarchy_io_flow.py \
  --top ibex_controller --target exc_cause_o \
  example/ibex_controller/preprocessed_code/ibex_controller_preprocessed_code.v
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

# 允许直接从任意路径运行脚本
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src3.verilog_static.api import analyze_verilog_files
from src3.verilog_static.ast_utils import get_params, get_ports
from src3.verilog_static.parser import get_modules, parse_verilog


@dataclass(frozen=True)
class InstEdge:
    parent_module: str
    child_module: str
    instance_name: str
    lineno: Optional[int]
    file: str


def _looks_like_module_file(vfile: str) -> bool:
    """
    轻量判断：文件里是否包含 module 定义。
    目的：跳过像 timescale.v / *defines.v 这种“只有指令/宏，没有 module”的文件，
    因为 PyVerilog 对它们做单文件 parse 时可能直接报 `at end of input`。
    """
    try:
        with open(vfile, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return False

    # 只要出现 `module <name>` 就认为是 module 文件（不做复杂注释剥离，够用）
    import re

    return re.search(r"(?m)^\s*module\b", text) is not None


def _iter_v_files(root_dir: str, recursive: bool) -> List[str]:
    root_dir = os.path.abspath(root_dir)
    vfiles: List[str] = []
    if recursive:
        for d, _, files in os.walk(root_dir):
            for fn in files:
                if fn.lower().endswith(".v"):
                    vfiles.append(os.path.join(d, fn))
    else:
        for fn in os.listdir(root_dir):
            if fn.lower().endswith(".v"):
                vfiles.append(os.path.join(root_dir, fn))
    vfiles.sort()
    return vfiles


def _iter_instances_in_moduledef(module_ast: Any) -> Iterable[Tuple[str, str, Optional[int]]]:
    """
    遍历 ModuleDef AST，产出 (instance_name, child_module_name, lineno)。
    """
    stack = [module_ast]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        cls = node.__class__.__name__
        if cls == "Function":
            # 跳过函数体内部
            continue
        if cls == "Instance":
            yield (getattr(node, "name", ""), getattr(node, "module", ""), getattr(node, "lineno", None))
        for c in node.children():
            stack.append(c)


def build_per_file_asts_and_callgraph(
    vfiles: List[str],
    include: List[str],
    define: List[str],
) -> Tuple[Dict[str, Any], nx.DiGraph, List[InstEdge], Dict[str, List[str]]]:
    """
    对每个文件单独 parse，返回：
    - file_ast: file -> AST
    - callgraph: moduleA -> moduleB（A 实例化 B）
    - inst_edges: 带 instance/line/file 的详细边
    - file_modules: file -> [module names defined in this file]
    """
    file_ast: Dict[str, Any] = {}
    callgraph = nx.DiGraph()
    inst_edges: List[InstEdge] = []
    file_modules: Dict[str, List[str]] = {}

    for vf in vfiles:
        # 跳过“纯 include/宏”文件（没有 module），否则单文件 parse 可能报 at end of input
        if not _looks_like_module_file(vf):
            file_modules[vf] = []
            continue
        pres = parse_verilog([vf], include=include, define=define)
        file_ast[vf] = pres.ast

        module_defs: Dict[str, Any] = {}
        module_instances: Dict[str, List[str]] = {}
        get_modules(pres.ast, module_defs, module_instances)
        file_modules[vf] = sorted(module_defs.keys())

        for parent, m_ast in module_defs.items():
            callgraph.add_node(parent)
            for inst_name, child_mod, lineno in _iter_instances_in_moduledef(m_ast):
                if not child_mod:
                    continue
                callgraph.add_node(child_mod)
                callgraph.add_edge(parent, child_mod)
                inst_edges.append(
                    InstEdge(
                        parent_module=parent,
                        child_module=child_mod,
                        instance_name=inst_name,
                        lineno=lineno,
                        file=vf,
                    )
                )

    return file_ast, callgraph, inst_edges, file_modules


def build_io_table_from_moduledefs(module_defs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 module_defs（module -> ModuleDef AST）构建 I/O table。
    """
    rows: List[Dict[str, Any]] = []
    for mname, mast in module_defs.items():
        params = get_params(mast, mname)
        ports = get_ports(mast, mname, params)

        for pname, width in ports["IPort"].items():
            rows.append({"module": mname, "port": pname, "dir": "input", "width": width})
        for pname, width in ports["OPort"].items():
            rows.append({"module": mname, "port": pname, "dir": "output", "width": width})

    # 排序：module -> dir -> port
    rows.sort(key=lambda r: (r["module"], r["dir"], r["port"]))
    return rows


def _find_target_node(dep_g: nx.DiGraph, target_signal: str) -> Optional[str]:
    """
    在 dep_g 节点中定位 target_signal。
    允许用户传 "sig" / "top.sig" / "sig[0]" 等；这里做最小的容错。
    """
    if target_signal in dep_g:
        return target_signal

    # 去掉 bit select，例如 a[3] -> a
    base = target_signal.split("[", 1)[0]
    if base in dep_g:
        return base

    # 如果用户传了 scope，但图里没这个 scope（或相反），这里不做太多猜测
    return None


def _collect_sinks_reachable(dep_g: nx.DiGraph, src: str) -> List[str]:
    reachable = set(nx.descendants(dep_g, src)) | {src}
    sinks = [n for n in reachable if dep_g.out_degree(n) == 0]
    sinks.sort()
    return sinks


def _collect_sources_reachable_reverse(dep_g: nx.DiGraph, src: str) -> List[str]:
    reachable = set(nx.ancestors(dep_g, src)) | {src}
    sources = [n for n in reachable if dep_g.in_degree(n) == 0]
    sources.sort()
    return sources


def enumerate_paths_downstream(
    dep_g: nx.DiGraph,
    src: str,
    *,
    max_depth: int = 25,
    max_paths: int = 200,
) -> List[List[str]]:
    """
    枚举 src -> sink 的所有简单路径（避免环）。
    方向：沿 dep_g 的 successors（u -> v 表示 v 依赖 u）。
    """
    sinks = set(_collect_sinks_reachable(dep_g, src))
    paths: List[List[str]] = []

    stack: List[Tuple[str, List[str], Set[str]]] = [(src, [src], {src})]
    while stack:
        node, path, seen = stack.pop()
        if node in sinks and node != src:
            paths.append(path)
            if len(paths) >= max_paths:
                break
            continue
        if len(path) >= max_depth:
            continue
        # 依赖图方向：u -> v 表示 v 依赖 u，因此下游传播看 successors
        for nxt in dep_g.successors(node):
            if nxt in seen:
                continue
            stack.append((nxt, path + [nxt], seen | {nxt}))

    return paths


def enumerate_paths_upstream(
    dep_g: nx.DiGraph,
    src: str,
    *,
    max_depth: int = 25,
    max_paths: int = 200,
) -> List[List[str]]:
    """
    枚举 src -> source 的所有简单路径（避免环）。
    方向：沿 dep_g 的 predecessors（追踪“src 由哪些信号决定/驱动”）。
    """
    sources = set(_collect_sources_reachable_reverse(dep_g, src))
    paths: List[List[str]] = []

    stack: List[Tuple[str, List[str], Set[str]]] = [(src, [src], {src})]
    while stack:
        node, path, seen = stack.pop()
        if node in sources and node != src:
            paths.append(path)
            if len(paths) >= max_paths:
                break
            continue
        if len(path) >= max_depth:
            continue
        for nxt in dep_g.predecessors(node):
            if nxt in seen:
                continue
            stack.append((nxt, path + [nxt], seen | {nxt}))

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Verilog: AST + 模块调用图 + I/O 表 + target_signal 数据流路径枚举")
    ap.add_argument("vfiles", nargs="*", help="输入 Verilog .v 文件（可多个）。也可用 --dir 批量扫描目录")
    ap.add_argument("--dir", dest="in_dir", default=None, help="输入目录：自动扫描目录内所有 .v（可配合 --recursive）")
    ap.add_argument("--recursive", action="store_true", help="配合 --dir 递归扫描子目录")
    ap.add_argument("--include", action="append", default=[], help="传给 PyVerilog 的 include 路径（可多次）")
    ap.add_argument("--define", action="append", default=[], help="传给 PyVerilog 的 define（可多次）")
    ap.add_argument("--top", dest="top_module", default=None, help="顶层模块名（不填则自动推断）")
    ap.add_argument("--target", dest="target_signal", required=True, help="顶层 target signal 名称（例如 exc_cause_o）")
    ap.add_argument("--max-depth", type=int, default=25, help="路径最大深度（防止图很大时爆炸）")
    ap.add_argument("--max-paths", type=int, default=200, help="最多打印/返回多少条路径")
    ap.add_argument(
        "--direction",
        choices=["downstream", "upstream", "both"],
        default="both",
        help="路径方向：downstream=谁依赖 target；upstream=target 由谁决定；both=两者都打印（默认）",
    )
    ap.add_argument(
        "--no-linking",
        action="store_true",
        help="不做跨模块 Linking（只分析当前输入文件里解析到的 module）。如果缺少子模块定义可先用它跑通。",
    )
    args = ap.parse_args()

    # 收集输入文件：支持“手动列文件”或“给目录自动扫描”
    vfiles: List[str] = []
    if args.in_dir:
        if args.vfiles:
            raise SystemExit("参数冲突：使用 --dir 时不要再额外传 vfiles 列表")
        vfiles = _iter_v_files(args.in_dir, recursive=bool(args.recursive))
        if not vfiles:
            raise SystemExit(f"目录里没有找到任何 .v 文件: {args.in_dir}")
    else:
        if not args.vfiles:
            raise SystemExit("请提供 .v 文件列表，或使用 --dir 指定目录")
        vfiles = [os.path.abspath(v) for v in args.vfiles]

    # 去重保持顺序
    vfiles = list(dict.fromkeys([os.path.abspath(v) for v in vfiles]))
    # PyVerilog 的 `include "xxx.v"` 只会在 preprocess_include 指定的目录中搜索；
    # 把被 include 的文件当作普通输入 vfile 传入并不能解决 include 查找问题。
    # 这里默认把每个输入文件所在目录加入 include 路径，避免常见 “Include file not found”。
    include_dirs = list(dict.fromkeys([os.path.abspath(p) for p in (args.include or [])] + [os.path.dirname(v) for v in vfiles]))
    # 仅把真正含 module 的文件作为“设计文件”参与全量分析；宏/指令文件仅用于 include 搜索
    design_vfiles = [vf for vf in vfiles if _looks_like_module_file(vf)]
    if not design_vfiles:
        print("未发现任何包含 `module` 定义的输入文件；请检查你传入的 .v 列表。")
    else:
        print(f"识别到设计文件（含 module）数量: {len(design_vfiles)}")
        for f in design_vfiles:
            print(f"- {os.path.basename(f)}")

    # 1) per-file AST + module instantiation callgraph
    _, callgraph, inst_edges, file_modules = build_per_file_asts_and_callgraph(design_vfiles, include_dirs, args.define)

    print("### 1) 模块调用关系图（谁实例化谁）")
    if callgraph.number_of_edges() == 0:
        print("（未发现任何 Instance，或输入文件中没有 module 定义）")
    else:
        for u, v in sorted(callgraph.edges()):
            print(f"- {u}  ->  {v}")

    print("\n（带 instance/line/file 的明细）")
    if not inst_edges:
        print("（无）")
    else:
        inst_edges_sorted = sorted(inst_edges, key=lambda e: (e.parent_module, e.child_module, e.file, e.lineno or -1, e.instance_name))
        for e in inst_edges_sorted:
            ln = "?" if e.lineno is None else str(e.lineno)
            print(f"- {e.parent_module} 实例化 {e.child_module}，instance={e.instance_name}，line={ln}，file={os.path.basename(e.file)}")

    print("\n（每个文件里定义了哪些 module）")
    for f, mods in sorted(file_modules.items(), key=lambda kv: kv[0]):
        print(f"- {os.path.basename(f)}: {', '.join(mods) if mods else '(无 module 定义，仅 include/宏文件，已跳过单文件 parse)'}")

    # 2) I/O table：为了全局一致性，这里对所有文件合并 parse 一次，拿到完整 module_defs
    try:
        r = analyze_verilog_files(
            design_vfiles,
            top_module=args.top_module,
            include=include_dirs,
            define=args.define,
            intermodular=(not args.no_linking),
            temporal_depth=1,  # 本脚本不依赖 COI 深度；只是为了构建 dep_g
        )
    except KeyError as e:
        missing = str(e).strip("'")
        print("\n### 解析/Linking 失败：缺少被实例化模块的定义")
        print(f"- 缺少 module: {missing}")
        print("- 这通常意味着你没把对应的 .v 源文件加入命令行输入列表（仅靠 `include` 不会把 module 定义补齐）。")
        print("- 解决方式：把缺的模块文件也加进来，例如：")
        print("  python3 -u example/verilog_hierarchy_io_flow.py --top ... --target ... a.v b.v <missing_module>.v")
        print("- 或者先加 `--no-linking` 只做单文件/单模块分析。")
        raise
    module_defs = r["module_defs"]
    io_rows = build_io_table_from_moduledefs(module_defs)

    print("\n### 2) I/O Table（module / port / dir / width）")
    if not io_rows:
        print("（无）")
    else:
        for row in io_rows:
            print(f"- {row['module']}.{row['port']}: {row['dir']} width={row['width']}")

    # 3) target_signal 数据流路径枚举（在跨模块 complete_dep_g 上）
    dep_g: nx.DiGraph = r["complete_dep_g"]
    top_module = r["top_module"]
    target = _find_target_node(dep_g, args.target_signal)

    print("\n### 3) target_signal 数据流路径（沿 dep_g successors）")
    print(f"- top_module: {top_module}")
    print(f"- target_signal: {args.target_signal}")

    if target is None:
        print("未在依赖图节点中找到 target_signal。可检查：")
        print("- 目标是否写错（大小写/下划线）")
        print("- 目标是否带了 bit select（例如 a[3]）可尝试只传 a")
        print("- 如果是跨模块信号，可能需要传作用域名（例如 u0.sig）")
        return

    print(f"- 匹配到图节点: {target}")
    if args.direction in {"downstream", "both"}:
        sinks = _collect_sinks_reachable(dep_g, target)
        print(f"- downstream 可达 sinks（out_degree==0）数量: {len(sinks)}")
        dpaths = enumerate_paths_downstream(dep_g, target, max_depth=args.max_depth, max_paths=args.max_paths)
        if not dpaths:
            print("（downstream：没有找到从 target 到 sink 的路径；可能 target 本身就是 sink，或图里只有环）")
        else:
            print(f"- downstream 枚举到路径条数: {len(dpaths)}（max_paths={args.max_paths}, max_depth={args.max_depth}）")
            for i, p in enumerate(dpaths, 1):
                print(f"[D{i}] " + " -> ".join(p))

    if args.direction in {"upstream", "both"}:
        sources = _collect_sources_reachable_reverse(dep_g, target)
        print(f"- upstream 可达 sources（in_degree==0）数量: {len(sources)}")
        upaths = enumerate_paths_upstream(dep_g, target, max_depth=args.max_depth, max_paths=args.max_paths)
        if not upaths:
            print("（upstream：没有找到从 target 到 source 的路径；可能 target 本身就是 source，或图里只有环）")
        else:
            print(f"- upstream 枚举到路径条数: {len(upaths)}（max_paths={args.max_paths}, max_depth={args.max_depth}）")
            for i, p in enumerate(upaths, 1):
                # 上游路径是 target -> predecessor -> ...；打印时用 '<-' 更直观
                print(f"[U{i}] " + " <- ".join(p))


if __name__ == "__main__":
    main()


