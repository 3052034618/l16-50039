#!/usr/bin/env python3
"""
随机分布采样引擎 - 命令行入口
================================

用法:
    python cli.py <分布名> [参数...] [选项]
    python cli.py batch <配置文件.json> [选项]

示例:
    # 生成 10000 个标准正态样本, 种子 42
    python cli.py normal --mu 0 --sigma 1 -n 10000 --seed 42

    # 生成泊松 λ=100 的样本并导出 CSV、报告和直方图
    python cli.py poisson --lam 100 -n 50000 --seed 777 \
        --export-csv poisson_100.csv \
        --export-report poisson_100.json \
        --export-histogram poisson_100_hist.csv \
        --histogram-bins 30

    # 批量运行多组实验
    python cli.py batch experiments.json \
        --summary-csv summary.csv \
        --summary-json summary.json

    # 查看所有支持的分布
    python cli.py --list

    # 查看某个分布的帮助
    python cli.py normal --help
"""

import argparse
import sys
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from sampling_engine import (
    UniformRNG,
    sample_uniform,
    sample_exponential,
    sample_normal_boxmuller,
    sample_poisson,
    sample_gamma,
    sample_beta,
    sample_binomial,
    sample_geometric,
    sample_bernoulli,
    batch_sample,
    compute_statistics,
    format_statistics_report,
    export_samples_csv,
    generate_report_json,
    save_report_json,
    compute_histogram,
    export_histogram_csv,
    load_batch_config,
    run_batch_experiments,
    format_batch_summary_table,
    export_batch_summary_csv,
    export_batch_summary_json,
)


# ============================================================================
# 分布元数据定义
# ============================================================================

DistributionSpec = Dict[str, Any]

DISTRIBUTIONS: Dict[str, DistributionSpec] = {
    "uniform": {
        "description": "连续均匀分布 U(a, b)",
        "sampler": sample_uniform,
        "params": [
            ("a", float, "下界 (默认 0.0)", 0.0),
            ("b", float, "上界 (默认 1.0)", 1.0),
        ],
        "theory": lambda p: (
            (p["a"] + p["b"]) / 2.0,
            (p["b"] - p["a"]) ** 2 / 12.0,
        ),
    },
    "exponential": {
        "description": "指数分布 Exp(λ)",
        "sampler": sample_exponential,
        "params": [
            ("lam", float, "率参数 λ > 0 (默认 1.0)", 1.0),
        ],
        "theory": lambda p: (1.0 / p["lam"], 1.0 / (p["lam"] ** 2)),
    },
    "normal": {
        "description": "正态分布 N(μ, σ²)  [Box-Muller]",
        "sampler": sample_normal_boxmuller,
        "params": [
            ("mu", float, "均值 μ (默认 0.0)", 0.0),
            ("sigma", float, "标准差 σ > 0 (默认 1.0)", 1.0),
        ],
        "theory": lambda p: (p["mu"], p["sigma"] ** 2),
    },
    "poisson": {
        "description": "泊松分布 Poisson(λ)  [分三档算法]",
        "sampler": sample_poisson,
        "params": [
            ("lam", float, "参数 λ > 0 (均值=方差=λ)", None),
        ],
        "theory": lambda p: (p["lam"], p["lam"]),
    },
    "gamma": {
        "description": "Gamma 分布 Gamma(shape, rate)  [Marsaglia-Tsang]",
        "sampler": sample_gamma,
        "params": [
            ("shape", float, "形状参数 α > 0", None),
            ("rate", float, "率参数 β > 0 (默认 1.0)", 1.0),
        ],
        "theory": lambda p: (p["shape"] / p["rate"], p["shape"] / (p["rate"] ** 2)),
    },
    "beta": {
        "description": "Beta 分布 Beta(α, β)  [基于 Gamma]",
        "sampler": sample_beta,
        "params": [
            ("alpha", float, "形状参数 α > 0", None),
            ("beta", float, "形状参数 β > 0", None),
        ],
        "theory": lambda p: (
            p["alpha"] / (p["alpha"] + p["beta"]),
            (p["alpha"] * p["beta"])
            / ((p["alpha"] + p["beta"]) ** 2 * (p["alpha"] + p["beta"] + 1)),
        ),
    },
    "binomial": {
        "description": "二项分布 Binomial(n, p)",
        "sampler": sample_binomial,
        "params": [
            ("n", int, "试验次数 n > 0", None),
            ("p", float, "成功概率 0 ≤ p ≤ 1", None),
        ],
        "theory": lambda p: (p["n"] * p["p"], p["n"] * p["p"] * (1.0 - p["p"])),
    },
    "geometric": {
        "description": "几何分布 Geometric(p)  [首次成功前失败次数]",
        "sampler": sample_geometric,
        "params": [
            ("p", float, "成功概率 0 < p ≤ 1", None),
        ],
        "theory": lambda p: ((1.0 - p["p"]) / p["p"], (1.0 - p["p"]) / (p["p"] ** 2)),
    },
    "bernoulli": {
        "description": "伯努利分布 Bernoulli(p)  [0/1 二值]",
        "sampler": sample_bernoulli,
        "params": [
            ("p", float, "成功概率 0 ≤ p ≤ 1", None),
        ],
        "theory": lambda p: (p["p"], p["p"] * (1.0 - p["p"])),
    },
}


# ============================================================================
# 命令行参数解析
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sampling-engine",
        description="随机分布采样引擎 - 从均匀随机数生成各种概率分布样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="支持的分布:\n"
        + "\n".join(f"  {name:>12} - {spec['description']}"
                    for name, spec in DISTRIBUTIONS.items()),
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有支持的分布并退出",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="命令",
        help="可用命令: 分布名 (如 normal, poisson) 或 batch",
    )

    # --- 单分布采样子命令 ---
    for dist_name, spec in DISTRIBUTIONS.items():
        sub = subparsers.add_parser(
            dist_name,
            help=spec["description"],
            description=spec["description"],
        )
        # 分布参数
        for pname, ptype, phelp, pdefault in spec["params"]:
            kwargs: Dict[str, Any] = {
                "type": ptype,
                "help": phelp,
            }
            if pdefault is not None:
                kwargs["default"] = pdefault
            else:
                kwargs["required"] = True
            sub.add_argument(f"--{pname}", **kwargs)

        # 通用选项
        sub.add_argument(
            "--num", "-n",
            type=int, default=10000,
            help="生成的样本数量 (默认 10000)",
        )
        sub.add_argument(
            "--seed", "-s",
            type=int, default=None,
            help="随机种子 (默认随机)",
        )
        sub.add_argument(
            "--export-csv",
            type=str, default=None, metavar="路径",
            help="将样本导出为 CSV 文件",
        )
        sub.add_argument(
            "--export-report",
            type=str, default=None, metavar="路径",
            help="将统计报告导出为 JSON 文件",
        )
        sub.add_argument(
            "--export-histogram",
            type=str, default=None, metavar="路径",
            help="将直方图分箱统计导出为 CSV 文件",
        )
        sub.add_argument(
            "--histogram-bins",
            type=int, default=20, metavar="N",
            help="直方图分箱数量 (默认 20)",
        )
        sub.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="静默模式, 不打印统计摘要",
        )
        sub.add_argument(
            "--no-index",
            action="store_true",
            help="导出 CSV 时不包含序号列",
        )

    # --- 批量实验子命令 ---
    batch_parser = subparsers.add_parser(
        "batch",
        help="批量运行多组实验 (从 JSON 配置文件)",
        description="从 JSON 配置文件批量运行多组采样实验, 自动生成汇总报告",
    )
    batch_parser.add_argument(
        "config",
        type=str,
        help="JSON 配置文件路径",
    )
    batch_parser.add_argument(
        "--summary-csv",
        type=str, default=None, metavar="路径",
        help="将多组实验汇总导出为 CSV",
    )
    batch_parser.add_argument(
        "--summary-json",
        type=str, default=None, metavar="路径",
        help="将多组实验汇总导出为 JSON",
    )
    batch_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式, 不打印进度和汇总表格",
    )
    batch_parser.add_argument(
        "--no-summary-table",
        action="store_true",
        help="不打印汇总表格 (但仍导出文件)",
    )

    return parser


# ============================================================================
# 单分布采样执行逻辑
# ============================================================================

def run_single_sampling(
    dist_name: str,
    params: Dict[str, Any],
    num_samples: int,
    seed: Optional[int],
) -> Tuple[List[float], Dict[str, Any], Optional[float], Optional[float]]:
    """执行单组采样并返回结果与理论值。"""
    spec = DISTRIBUTIONS[dist_name]
    sampler = spec["sampler"]

    rng = UniformRNG(seed=seed)
    samples = batch_sample(sampler, num_samples, rng=rng, **params)

    theory_mean, theory_var = spec["theory"](params)

    return samples, params, theory_mean, theory_var


def handle_single_command(args: argparse.Namespace) -> int:
    """处理单分布采样命令。"""
    dist_name: str = args.command
    spec = DISTRIBUTIONS[dist_name]

    # 收集分布参数
    params: Dict[str, Any] = {}
    for pname, ptype, _, pdefault in spec["params"]:
        val = getattr(args, pname)
        params[pname] = val

    num_samples = args.num
    seed = args.seed

    if num_samples <= 0:
        print(f"错误: 样本数量必须为正整数, 得到 {num_samples}", file=sys.stderr)
        return 2

    # 执行采样
    try:
        samples, params_used, theory_mean, theory_var = run_single_sampling(
            dist_name, params, num_samples, seed
        )
    except ValueError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"采样时发生错误: {type(e).__name__}: {e}", file=sys.stderr)
        return 4

    # 计算统计
    try:
        stats = compute_statistics(samples)
    except Exception as e:
        print(f"统计计算错误: {e}", file=sys.stderr)
        return 5

    # 打印摘要
    if not args.quiet:
        title = f"{dist_name} 分布 {params_used}"
        if seed is not None:
            title += f"  (seed={seed})"
        report_str = format_statistics_report(
            stats,
            title=title,
            theoretical_mean=theory_mean,
            theoretical_var=theory_var,
        )
        print("\n" + report_str + "\n")

    exported_paths: List[str] = []

    # 导出样本 CSV
    if args.export_csv:
        try:
            csv_path = export_samples_csv(
                samples,
                args.export_csv,
                include_index=not args.no_index,
            )
            exported_paths.append(("样本 CSV", csv_path))
            if not args.quiet:
                print(f"✓ 样本已导出到: {csv_path}")
        except Exception as e:
            print(f"CSV 导出失败: {e}", file=sys.stderr)
            return 6

    # 导出 JSON 报告
    if args.export_report:
        try:
            report = generate_report_json(
                dist_name,
                params_used,
                stats,
                seed=seed,
                theoretical_mean=theory_mean,
                theoretical_var=theory_var,
            )
            report_path = save_report_json(report, args.export_report)
            exported_paths.append(("统计报告 JSON", report_path))
            if not args.quiet:
                print(f"✓ 统计报告已导出到: {report_path}")
        except Exception as e:
            print(f"报告导出失败: {e}", file=sys.stderr)
            return 7

    # 导出直方图 CSV
    if args.export_histogram:
        try:
            hist = compute_histogram(samples, bins=args.histogram_bins)
            hist_path = export_histogram_csv(hist, args.export_histogram)
            exported_paths.append(("直方图 CSV", hist_path))
            if not args.quiet:
                print(f"✓ 直方图统计已导出到: {hist_path}")
        except Exception as e:
            print(f"直方图导出失败: {e}", file=sys.stderr)
            return 8

    if not args.quiet and exported_paths:
        print()

    return 0


# ============================================================================
# 批量实验执行逻辑
# ============================================================================

def handle_batch_command(args: argparse.Namespace) -> int:
    """处理批量实验命令。"""
    config_path = args.config

    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}", file=sys.stderr)
        return 10

    # 加载配置
    try:
        config = load_batch_config(config_path)
    except ValueError as e:
        print(f"配置文件错误: {e}", file=sys.stderr)
        return 11
    except Exception as e:
        print(f"加载配置文件失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 12

    # 运行批量实验
    try:
        batch_result = run_batch_experiments(config, quiet=args.quiet)
    except ValueError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        return 13
    except Exception as e:
        print(f"批量实验失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 14

    summary = batch_result["summary"]
    output_dir = batch_result["output_dir"]

    # 打印汇总表格
    if not args.quiet and not args.no_summary_table:
        print("\n" + "=" * 80)
        print("  批量实验汇总报告")
        print("=" * 80)
        print(format_batch_summary_table(summary))
        print()

    # 导出汇总 CSV
    if args.summary_csv:
        try:
            csv_path = export_batch_summary_csv(summary, args.summary_csv)
            if not args.quiet:
                print(f"✓ 汇总 CSV 已导出到: {csv_path}")
        except Exception as e:
            print(f"汇总 CSV 导出失败: {e}", file=sys.stderr)
            return 15

    # 导出汇总 JSON
    if args.summary_json:
        try:
            json_path = export_batch_summary_json(summary, args.summary_json)
            if not args.quiet:
                print(f"✓ 汇总 JSON 已导出到: {json_path}")
        except Exception as e:
            print(f"汇总 JSON 导出失败: {e}", file=sys.stderr)
            return 16

    if not args.quiet and (args.summary_csv or args.summary_json):
        print()

    if not args.quiet:
        print(f"所有实验结果已输出到: {output_dir}")
        print()

    return 0


# ============================================================================
# 主入口
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("\n支持的分布:\n")
        for name, spec in DISTRIBUTIONS.items():
            print(f"  {name:>12} - {spec['description']}")
        print()
        print("查看某分布的详细参数: python cli.py <分布名> --help")
        print("批量实验帮助:       python cli.py batch --help\n")
        return 0

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "batch":
        return handle_batch_command(args)
    else:
        return handle_single_command(args)


if __name__ == "__main__":
    sys.exit(main())
