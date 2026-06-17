"""
使用示例与分布验证
==================
演示 sampling_engine 模块的各种用法, 包含边界场景验证。
"""

import math
import time
from collections import Counter
from sampling_engine import (
    UniformRNG,
    sample_uniform,
    sample_exponential,
    sample_normal_boxmuller,
    sample_normal_pair_boxmuller,
    sample_poisson,
    AliasSampler,
    sample_discrete_alias,
    sample_weighted,
    sample_bernoulli,
    sample_multinomial,
    batch_sample,
)


def print_separator(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def pass_fail(ok: bool) -> str:
    return "✓ PASS" if ok else "✗ FAIL"


# ============================================================================
# 一、基础分布验证
# ============================================================================

def verify_uniform_distribution() -> None:
    print_separator("验证 1: 均匀分布卡方检验")
    rng = UniformRNG(seed=12345)
    n = 100000
    n_bins = 10
    bins = [0] * n_bins

    for _ in range(n):
        u = rng.random()
        idx = int(u * n_bins)
        if idx == n_bins:
            idx = n_bins - 1
        bins[idx] += 1

    expected = n / n_bins
    chi_sq = sum((c - expected) ** 2 / expected for c in bins)
    passed = chi_sq < 16.92  # 自由度 9, 0.95 分位数
    print(f"  卡方统计量 = {chi_sq:.3f} (阈值 ≈ 16.92)  {pass_fail(passed)}")


def verify_inverse_transform() -> None:
    print_separator("验证 2: 逆变换采样 (均匀 + 指数)")

    # 指数分布 Exp(2)
    lam = 2.0
    n = 30000
    rng = UniformRNG(seed=1)
    samples = batch_sample(sample_exponential, n, lam=lam, rng=rng)
    m = sum(samples) / n
    v = sum((x - m) ** 2 for x in samples) / n
    ok_m = abs(m - 1 / lam) < 0.02
    ok_v = abs(v - 1 / (lam ** 2)) < 0.02
    print(f"  指数 Exp({lam}): 均值={m:.4f}(期望{1/lam:.4f}) {pass_fail(ok_m)},  "
          f"方差={v:.4f}(期望{1/(lam**2):.4f}) {pass_fail(ok_v)}")


def verify_normal() -> None:
    print_separator("验证 3: 正态分布 Box-Muller (N(10, 3^2))")
    mu, sigma = 10.0, 3.0
    n = 60000
    rng = UniformRNG(seed=99)
    samples = batch_sample(sample_normal_boxmuller, n, mu=mu, sigma=sigma, rng=rng)

    m = sum(samples) / n
    s = math.sqrt(sum((x - m) ** 2 for x in samples) / n)
    in2s = sum(1 for x in samples if abs(x - mu) <= 2 * sigma) / n * 100

    ok_m = abs(m - mu) < 0.05
    ok_s = abs(s - sigma) < 0.05
    ok_2s = abs(in2s - 95.45) < 0.5
    print(f"  均值 = {m:.4f}(期望 {mu}) {pass_fail(ok_m)}")
    print(f"  标准差 = {s:.4f}(期望 {sigma}) {pass_fail(ok_s)}")
    print(f"  2σ 比例 = {in2s:.2f}%(期望 95.45%) {pass_fail(ok_2s)}")

    # 独立性验证: Box-Muller 成对样本的相关系数
    pairs = [sample_normal_pair_boxmuller(mu=0, sigma=1, rng=UniformRNG(i))
             for i in range(2000)]
    corr = sum(a * b for a, b in pairs) / len(pairs)
    ok_corr = abs(corr) < 0.1
    print(f"  成对样本相关系数 = {corr:.4f} (应接近 0) {pass_fail(ok_corr)}")


# ============================================================================
# 二、泊松分布 (重点: 大 λ 性能验证)
# ============================================================================

def verify_poisson_large_lambda() -> None:
    print_separator("验证 4: 泊松分布大 λ 性能与精度 (λ = 30, 100, 1000)")

    lambdas = [30.0, 100.0, 1000.0]
    n_samples = 15000

    print(f"\n  样本量 N = {n_samples}, 超时阈值: 每项 3 秒\n")
    print(f"  {'λ':>6} | {'耗时(s)':>8} | {'样本均值':>9} | {'样本方差':>9} | "
          f"{'均值偏差%':>9} | {'方差偏差%':>9} | 结果")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}--{'-'*6}")

    all_ok = True
    for lam in lambdas:
        rng = UniformRNG(seed=777 + int(lam))
        t0 = time.perf_counter()
        samples = batch_sample(sample_poisson, n_samples, lam=lam, rng=rng)
        elapsed = time.perf_counter() - t0

        m = sum(samples) / n_samples
        v = sum((x - m) ** 2 for x in samples) / n_samples

        dev_m = (m - lam) / lam * 100
        dev_v = (v - lam) / lam * 100

        ok_time = elapsed < 3.0
        ok_m = abs(dev_m) < 5.0
        ok_v = abs(dev_v) < 10.0
        ok = ok_time and ok_m and ok_v
        if not ok:
            all_ok = False

        flag = pass_fail(ok)
        if not ok_time:
            flag += " (超时!)"

        print(f"  {lam:>6.0f} | {elapsed:>8.3f} | {m:>9.3f} | {v:>9.3f} | "
              f"{dev_m:>+8.2f}% | {dev_v:>+8.2f}% | {flag}")

    print(f"\n  大 λ 泊松总体: {pass_fail(all_ok)}")


# ============================================================================
# 三、离散分布 - 非法输入边界验证
# ============================================================================

def verify_alias_input_validation() -> None:
    print_separator("验证 5: 离散分布非法输入 -> 立即报错")

    cases = [
        ("空列表",            [],
         "概率分布不能为空列表"),
        ("包含负值 [0.3, -0.1, 0.8]", [0.3, -0.1, 0.8],
         "概率/权重不能为负值"),
        ("全零权重 [0, 0, 0]", [0.0, 0.0, 0.0],
         "所有概率/权重均为 0"),
        ("混合零与负值 [0, -5, 0]", [0, -5, 0],
         "概率/权重不能为负值"),
    ]

    print()
    all_ok = True
    for name, probs, expected_msg in cases:
        try:
            AliasSampler(probs)
            print(f"  ✗ [{name}] -> 未抛异常 (预期: {expected_msg!r})")
            all_ok = False
        except ValueError as e:
            got = str(e)
            matched = expected_msg in got
            if matched:
                print(f"  ✓ [{name}] -> ValueError: {got}")
            else:
                print(f"  ✗ [{name}] -> ValueError: {got!r} (未包含 {expected_msg!r})")
                all_ok = False
        except Exception as e:
            print(f"  ✗ [{name}] -> {type(e).__name__}: {e} (预期 ValueError)")
            all_ok = False

    # ---- 合法但未归一化的权重: 应该正常工作 ----
    print(f"\n  --- 合法但未归一化权重: 应该正常工作 ---")
    unnormalized = [1, 2, 0, 3, 4]   # 和 = 10
    labels = ["A", "B", "C", "D", "E"]
    expected_probs = [0.1, 0.2, 0.0, 0.3, 0.4]

    try:
        sampler = AliasSampler(unnormalized, rng=UniformRNG(seed=5))
        n = 50000
        counts = Counter(labels[sampler.sample()] for _ in range(n))
        ok_unnorm = True
        print(f"    输入权重 = {unnormalized}  (未归一化, 有 0)")
        print(f"    {'类':>3}  {'观测频率':>8}  {'理论概率':>8}  {'偏差%':>8}")
        for label, exp_p in zip(labels, expected_probs):
            obs = counts.get(label, 0) / n
            dev = (obs - exp_p) * 100
            mark = ""
            if exp_p == 0:
                if obs != 0:
                    ok_unnorm = False
                    mark = " ✗ (零权重被抽中!)"
            else:
                if abs(dev / (exp_p * 100)) > 0.1:  # 相对偏差 < 10%
                    ok_unnorm = False
                    mark = " ✗"
            print(f"    {label:>3}  {obs:>8.4f}  {exp_p:>8.3f}  {dev:>+7.2f}%{mark}")
        print(f"    未归一化权重抽样: {pass_fail(ok_unnorm)}")
        all_ok = all_ok and ok_unnorm
    except Exception as e:
        print(f"    ✗ 未归一化权重抛异常: {type(e).__name__}: {e}")
        all_ok = False

    print(f"\n  非法 / 边界输入总体: {pass_fail(all_ok)}")


# ============================================================================
# 四、加权抽样 - 零权重边界验证
# ============================================================================

def verify_weighted_zero_weight() -> None:
    print_separator("验证 6: 加权抽样 - 零权重不被抽中 + 正权重不足清晰报错")

    all_ok = True

    # ---- 无放回: 零权重物品永远不该被抽中 ----
    print("\n  --- 无放回: 零权重物品不应该出现在结果中 ---")
    items = ["A", "B", "C", "D", "E"]
    weights = [1, 0, 2, 0, 3]  # B 和 D 权重为 0
    forbidden = {items[i] for i, w in enumerate(weights) if w == 0}
    allowed = {items[i] for i, w in enumerate(weights) if w > 0}
    n_tests = 500
    hit_zero = 0
    rng = UniformRNG(seed=42)
    for _ in range(n_tests):
        result = sample_weighted(items, weights, k=3, replace=False, rng=rng)
        if any(x in forbidden for x in result):
            hit_zero += 1
    ok_noreplace_zero = hit_zero == 0 and all(
        set(sample_weighted(items, weights, k=3, replace=False, rng=rng)) <= allowed
        for _ in range(50)
    )
    print(f"    物品 = {items}")
    print(f"    权重 = {weights}  (零权重: {sorted(forbidden)})")
    print(f"    {n_tests} 次独立抽样, 零权重被抽中的次数 = {hit_zero}")
    print(f"    零权重未被抽中: {pass_fail(ok_noreplace_zero)}")
    all_ok = all_ok and ok_noreplace_zero

    # ---- 无放回: 正权重物品不够抽 -> 清晰报错 ----
    print("\n  --- 无放回: 正权重物品不足时清晰报错 ---")
    cases = [
        ("需要 3 个但正权重只有 2 个",
         ["A", "B", "C", "D"], [1, 0, 1, 0], 3,
         "正权重物品数量不足"),
        ("需要 2 个但全部零权重",
         ["X", "Y", "Z"], [0, 0, 0], 2,
         "正权重物品数量不足"),
    ]
    for name, its, wts, k, expected_msg in cases:
        try:
            sample_weighted(its, wts, k=k, replace=False, rng=UniformRNG(1))
            print(f"    ✗ [{name}] -> 未抛异常 (预期: {expected_msg!r})")
            all_ok = False
        except ValueError as e:
            matched = expected_msg in str(e)
            if matched:
                print(f"    ✓ [{name}] -> ValueError: {e}")
            else:
                print(f"    ✗ [{name}] -> ValueError: {e!r} (未包含 {expected_msg!r})")
                all_ok = False
        except Exception as e:
            print(f"    ✗ [{name}] -> {type(e).__name__}: {e}")
            all_ok = False

    # ---- 有放回: 未归一化 + 零权重 (别名方法覆盖) ----
    print("\n  --- 有放回: 零权重不应被抽中 (通过别名方法) ---")
    rng = UniformRNG(seed=101)
    n = 30000
    result = sample_weighted(items, weights, k=n, replace=True, rng=rng)
    counts = Counter(result)
    ok_replace_zero = not any(counts.get(x, 0) > 0 for x in forbidden)
    print(f"    物品 = {items}, 权重 = {weights}")
    print(f"    有放回抽样 {n} 次: 零权重物品计数 = "
          f"{ {x: counts.get(x, 0) for x in sorted(forbidden)} }")
    # 额外验证: 允许的物品比例符合预期
    total_pos = sum(w for w in weights if w > 0)
    ok_ratio = True
    for label, w in zip(items, weights):
        if w > 0:
            exp = w / total_pos
            obs = counts.get(label, 0) / n
            if abs(obs - exp) / exp > 0.15:
                ok_ratio = False
                print(f"      警告: {label} 观测 {obs:.4f} vs 期望 {exp:.4f} 偏差过大")
    print(f"    零权重未被抽中: {pass_fail(ok_replace_zero)}  |  "
          f"比例符合: {pass_fail(ok_ratio)}")
    all_ok = all_ok and ok_replace_zero and ok_ratio

    print(f"\n  加权抽样边界总体: {pass_fail(all_ok)}")


# ============================================================================
# 五、性能基准 (大参数泊松单独测)
# ============================================================================

def benchmark_poisson_only() -> None:
    print_separator("性能基准: 各分布抽样速度 (N = 100k)")
    n = 100000
    rng = UniformRNG(seed=123)

    tests = [
        ("均匀 U(0,1)",           lambda: rng.random()),
        ("指数 Exp(1)",           lambda: sample_exponential(1.0, rng=rng)),
        ("正态 N(0,1)",           lambda: sample_normal_boxmuller(0, 1, rng=rng)),
        ("泊松 λ=5 (Knuth)",      lambda: sample_poisson(5.0, rng=rng)),
        ("泊松 λ=30 (近似算法)",  lambda: sample_poisson(30.0, rng=rng)),
        ("泊松 λ=100 (近似算法)", lambda: sample_poisson(100.0, rng=rng)),
        ("泊松 λ=1000 (近似算法)",lambda: sample_poisson(1000.0, rng=rng)),
    ]

    print(f"\n  {'分布':<24} {'耗时(s)':>8} {'速度(K/s)':>10}")
    print(f"  {'-'*24} {'-'*8} {'-'*10}")
    for name, fn in tests:
        start = time.perf_counter()
        for _ in range(n):
            fn()
        elapsed = time.perf_counter() - start
        rate = n / elapsed / 1000
        print(f"  {name:<24} {elapsed:>8.4f} {rate:>10.0f}")

    print("\n  别名方法 (已预构建):")
    probs = [0.1] * 10
    sampler = AliasSampler(probs, rng=rng)
    start = time.perf_counter()
    for _ in range(n):
        sampler.sample()
    elapsed = time.perf_counter() - start
    rate = n / elapsed / 1000
    print(f"  {'10 类离散分布采样':<24} {elapsed:>8.4f} {rate:>10.0f}")


# ============================================================================
# main
# ============================================================================

def main() -> None:
    print("\n" + "▓" * 72)
    print("▓" + " " * 70 + "▓")
    print("▓        随机分布采样引擎 - 验证 (含大参数泊松 & 边界场景)         ▓")
    print("▓" + " " * 70 + "▓")
    print("▓" * 72)

    # 基础分布
    verify_uniform_distribution()
    verify_inverse_transform()
    verify_normal()

    # 重点: 大 λ 泊松
    verify_poisson_large_lambda()

    # 边界: 非法输入
    verify_alias_input_validation()

    # 边界: 零权重
    verify_weighted_zero_weight()

    # 性能
    benchmark_poisson_only()

    print("\n" + "=" * 72)
    print("  全部验证完成 (请查看上方每项的 ✓/✗ 标记)")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
