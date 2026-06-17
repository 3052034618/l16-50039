"""
使用示例与分布验证
==================
演示 sampling_engine 模块的各种用法, 并通过统计检验验证采样质量。
"""

import math
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
    sample_beta,
    batch_sample,
    inverse_transform_sample,
)


def print_separator(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_uniform_rng() -> None:
    """演示均匀随机数生成器。"""
    print_separator("一、均匀随机数生成器 (UniformRNG)")
    rng = UniformRNG(seed=42)

    print("\n[0, 1) 区间均匀随机数 (前 5 个):")
    for i in range(5):
        print(f"  U{i+1} = {rng.random():.6f}")

    print("\n(0, 1) 区间严格大于 0 的随机数 (前 5 个):")
    for i in range(5):
        print(f"  U{i+1} = {rng.random_open():.6f}")

    print("\n[1, 10] 区间随机整数 (前 10 个):")
    print(" ", [rng.randint(1, 10) for _ in range(10)])


def verify_uniform_distribution() -> None:
    """验证均匀分布: 分箱统计。"""
    print_separator("验证: [0,1) 均匀分布的分箱统计")
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
    print(f"\n样本数 N = {n}, 分箱数 = {n_bins}, 每箱期望 = {expected:.0f}")
    chi_sq = 0.0
    for i, count in enumerate(bins):
        interval = f"[{i/n_bins:.1f}, {(i+1)/n_bins:.1f})"
        dev = (count - expected) / expected * 100
        chi_sq += (count - expected) ** 2 / expected
        print(f"  区间 {interval}: {count:>6d}  (偏差 {dev:+.2f}%)")
    print(f"\n  卡方统计量: {chi_sq:.3f}  (自由度 9, 0.95 分位数 ≈ 16.92)")
    print(f"  {'✓ 通过' if chi_sq < 16.92 else '✗ 未通过'}卡方检验")


def demo_inverse_transform() -> None:
    """演示逆变换采样框架。"""
    print_separator("二、逆变换采样 (Inverse Transform Sampling)")

    print("\n[示例 1] 连续均匀分布 U(a, b)")
    a, b = 2.0, 5.0
    samples = batch_sample(sample_uniform, 10000, a=a, b=b)
    mean_emp = sum(samples) / len(samples)
    var_emp = sum((x - mean_emp) ** 2 for x in samples) / len(samples)
    mean_exp = (a + b) / 2
    var_exp = (b - a) ** 2 / 12
    print(f"  参数: a={a}, b={b}")
    print(f"  样本均值 = {mean_emp:.4f}  (理论 = {mean_exp:.4f})")
    print(f"  样本方差 = {var_emp:.4f}  (理论 = {var_exp:.4f})")

    print("\n[示例 2] 指数分布 Exp(λ)")
    lam = 2.0
    samples = batch_sample(sample_exponential, 10000, lam=lam)
    mean_emp = sum(samples) / len(samples)
    var_emp = sum((x - mean_emp) ** 2 for x in samples) / len(samples)
    mean_exp = 1.0 / lam
    var_exp = 1.0 / (lam ** 2)
    print(f"  参数: λ = {lam}")
    print(f"  样本均值 = {mean_emp:.4f}  (理论 = {mean_exp:.4f})")
    print(f"  样本方差 = {var_emp:.4f}  (理论 = {var_exp:.4f})")

    print("\n[示例 3] 使用通用框架自定义逆变换采样 (指数分布)")
    inv_cdf = lambda u, lam=lam: -math.log(u) / lam
    samples = [inverse_transform_sample(inv_cdf) for _ in range(10000)]
    mean_emp = sum(samples) / len(samples)
    print(f"  样本均值 = {mean_emp:.4f}  (理论 = {1/lam:.4f})  ✓")


def demo_normal_boxmuller() -> None:
    """演示正态分布 Box-Muller 采样。"""
    print_separator("三、正态分布采样 (Box-Muller 方法)")

    mu, sigma = 10.0, 3.0
    n = 100000

    print(f"\n采样 N({mu}, {sigma}^2), 样本数 N = {n}")

    rng = UniformRNG(seed=99)
    samples = batch_sample(sample_normal_boxmuller, n, mu=mu, sigma=sigma, rng=rng)

    mean_emp = sum(samples) / n
    var_emp = sum((x - mean_emp) ** 2 for x in samples) / n
    std_emp = math.sqrt(var_emp)

    print(f"  样本均值 = {mean_emp:.4f}  (理论 = {mu:.4f})")
    print(f"  样本标准差 = {std_emp:.4f}  (理论 = {sigma:.4f})")
    print(f"  样本方差 = {var_emp:.4f}  (理论 = {sigma**2:.4f})")

    within_1sigma = sum(1 for x in samples if abs(x - mu) <= sigma) / n * 100
    within_2sigma = sum(1 for x in samples if abs(x - mu) <= 2 * sigma) / n * 100
    within_3sigma = sum(1 for x in samples if abs(x - mu) <= 3 * sigma) / n * 100
    print(f"\n  经验规则验证 (正态分布):")
    print(f"    1σ 内比例: {within_1sigma:.2f}%  (理论 68.27%)")
    print(f"    2σ 内比例: {within_2sigma:.2f}%  (理论 95.45%)")
    print(f"    3σ 内比例: {within_3sigma:.2f}%  (理论 99.73%)")

    print("\n[演示] 一次生成两个独立正态样本:")
    z1, z2 = sample_normal_pair_boxmuller(mu=0, sigma=1, rng=UniformRNG(seed=1))
    print(f"  Z1 = {z1:.6f},  Z2 = {z2:.6f}")
    pairs = [sample_normal_pair_boxmuller(mu=0, sigma=1) for _ in range(1000)]
    corr = sum(a * b for a, b in pairs) / len(pairs)
    print(f"  1000 对样本的相关系数 ≈ {corr:.4f}  (应接近 0, 表示独立)")


def demo_poisson() -> None:
    """演示泊松分布采样。"""
    print_separator("四、泊松分布采样")

    for lam in [1.0, 5.0, 20.0, 100.0]:
        n = 50000
        rng = UniformRNG(seed=777)
        samples = batch_sample(sample_poisson, n, lam=lam, rng=rng)

        mean_emp = sum(samples) / n
        var_emp = sum((x - mean_emp) ** 2 for x in samples) / n

        print(f"\n  λ = {lam:>5.1f}  |  样本均值 = {mean_emp:>7.3f}  (理论 {lam})"
              f"  |  样本方差 = {var_emp:>7.3f}  (理论 {lam})")

    print("\n[示例] λ = 3 的泊松分布频数对比:")
    lam = 3.0
    n = 200000
    rng = UniformRNG(seed=42)
    samples = batch_sample(sample_poisson, n, lam=lam, rng=rng)
    counts = Counter(samples)

    def poisson_pmf(k, lam):
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    print(f"  {'k':>3}  {'观测频率':>10}  {'理论概率':>10}  {'偏差':>10}")
    for k in range(0, 10):
        obs_freq = counts.get(k, 0) / n
        theory_p = poisson_pmf(k, lam)
        dev = (obs_freq - theory_p) * 100
        print(f"  {k:>3}  {obs_freq:>10.4f}  {theory_p:>10.4f}  {dev:>+9.4f}%")


def demo_alias_method() -> None:
    """演示别名方法。"""
    print_separator("五、离散分布别名方法 (Alias Method)")

    probs = [0.1, 0.2, 0.05, 0.3, 0.15, 0.2]
    labels = ["A", "B", "C", "D", "E", "F"]

    print(f"\n目标分布:")
    for label, p in zip(labels, probs):
        print(f"  P({label}) = {p:.3f}")

    sampler = AliasSampler(probs, rng=UniformRNG(seed=101))
    print(f"\n别名表构建结果:")
    print(f"  prob   = {[f'{p:.4f}' for p in sampler.prob]}")
    print(f"  alias  = {sampler.alias}")

    n = 100000
    counts = Counter()
    for _ in range(n):
        idx = sampler.sample()
        counts[labels[idx]] += 1

    print(f"\n采样 {n} 次后的频率统计:")
    print(f"  {'类别':>5}  {'观测频率':>10}  {'理论概率':>10}  {'偏差':>10}")
    for label, p in zip(labels, probs):
        obs = counts.get(label, 0) / n
        dev = (obs - p) * 100
        print(f"  {label:>5}  {obs:>10.4f}  {p:>10.3f}  {dev:>+9.4f}%")

    print("\n[便捷函数] sample_discrete_alias (单次采样):")
    rng = UniformRNG(seed=5)
    print(" ", [labels[sample_discrete_alias(probs, rng=rng)] for _ in range(10)])


def demo_weighted_sampling() -> None:
    """演示加权随机抽样。"""
    print_separator("六、加权随机抽样")

    items = ["苹果", "香蕉", "橙子", "葡萄", "西瓜"]
    weights = [1, 2, 1, 3, 1]

    print(f"\n物品:   {items}")
    print(f"权重:   {weights}")

    print("\n--- 有放回加权抽样 (10000 次) ---")
    rng = UniformRNG(seed=2024)
    samples = sample_weighted(items, weights, k=10000, replace=True, rng=rng)
    counts = Counter(samples)
    total_w = sum(weights)
    print(f"  {'物品':>5}  {'观测频率':>10}  {'理论概率':>10}")
    for item, w in zip(items, weights):
        obs = counts.get(item, 0) / 10000
        theory = w / total_w
        print(f"  {item:>5}  {obs:>10.4f}  {theory:>10.4f}")

    print("\n--- 无放回加权抽样 (抽 3 个, 重复 10 次) ---")
    rng = UniformRNG(seed=88)
    for i in range(10):
        result = sample_weighted(items, weights, k=3, replace=False, rng=rng)
        print(f"  第 {i+1:>2} 次: {result}")


def demo_other_distributions() -> None:
    """演示其他辅助分布。"""
    print_separator("七、其他分布")

    print("\n[伯努利分布] p = 0.3:")
    rng = UniformRNG(seed=50)
    samples = [sample_bernoulli(0.3, rng=rng) for _ in range(10000)]
    print(f"  样本均值 = {sum(samples)/len(samples):.4f}  (理论 = 0.3000)")

    print("\n[多项分布] 100 次试验, 概率 [0.2, 0.3, 0.5]:")
    rng = UniformRNG(seed=60)
    counts = sample_multinomial(100, [0.2, 0.3, 0.5], rng=rng)
    print(f"  各类别计数: {counts}  (期望 [20, 30, 50])")

    print("\n[Beta 分布] α=2, β=5:")
    rng = UniformRNG(seed=70)
    samples = [sample_beta(2, 5, rng=rng) for _ in range(10000)]
    mean_emp = sum(samples) / len(samples)
    mean_exp = 2 / (2 + 5)
    print(f"  样本均值 = {mean_emp:.4f}  (理论 = {mean_exp:.4f})")


def benchmark_performance() -> None:
    """性能基准测试。"""
    print_separator("八、性能基准测试")
    import time

    n = 100000
    rng = UniformRNG(seed=123)

    tests = [
        ("均匀分布 U(0,1)", lambda: rng.random()),
        ("指数分布 Exp(1)", lambda: sample_exponential(1.0, rng=rng)),
        ("正态分布 N(0,1)", lambda: sample_normal_boxmuller(0, 1, rng=rng)),
        ("泊松分布 λ=5", lambda: sample_poisson(5.0, rng=rng)),
        ("泊松分布 λ=100", lambda: sample_poisson(100.0, rng=rng)),
    ]

    print(f"\n采样 {n} 次的耗时对比:")
    for name, fn in tests:
        start = time.perf_counter()
        for _ in range(n):
            fn()
        elapsed = time.perf_counter() - start
        rate = n / elapsed / 1000
        print(f"  {name:<20}: {elapsed:.4f} 秒  ({rate:.0f} K samples/sec)")

    print("\n别名方法性能:")
    probs = [0.1] * 10
    sampler = AliasSampler(probs, rng=rng)
    start = time.perf_counter()
    for _ in range(n):
        sampler.sample()
    elapsed = time.perf_counter() - start
    rate = n / elapsed / 1000
    print(f"  10 类离散分布 (已预构建): {elapsed:.4f} 秒  ({rate:.0f} K samples/sec)")


def main() -> None:
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█           随机分布采样引擎 - 使用示例与分布验证              █")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    demo_uniform_rng()
    verify_uniform_distribution()
    demo_inverse_transform()
    demo_normal_boxmuller()
    demo_poisson()
    demo_alias_method()
    demo_weighted_sampling()
    demo_other_distributions()
    benchmark_performance()

    print("\n" + "=" * 70)
    print("  所有演示完成 ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
