"""
随机分布采样引擎
================
提供从均匀随机数生成各种概率分布样本的完整实现。

包含:
- 高质量均匀随机数生成器封装
- 逆变换采样通用框架
- 正态分布 (Box-Muller)
- 指数分布
- 泊松分布
- 离散分布别名方法 (Alias Method)
- 加权随机抽样 (Weighted Random Sampling)
"""

import math
import random
import csv
import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from collections import deque

T = TypeVar("T")


# ============================================================================
# 一、均匀随机数生成器
# ============================================================================

class UniformRNG:
    """
    高质量均匀随机数生成器封装。

    基于 Python 标准库的 random 模块 (底层使用 Mersenne Twister MT19937)。
    MT19937 的周期为 2^19937 - 1, 远超过大多数应用场景的需求。
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._normal_cache: deque = deque()

    def random(self) -> float:
        """返回 [0, 1) 区间的均匀随机浮点数。"""
        return self._rng.random()

    def random_open(self) -> float:
        """返回 (0, 1) 区间的均匀随机浮点数 (严格大于 0)。"""
        u = self._rng.random()
        while u == 0.0:
            u = self._rng.random()
        return u

    def randint(self, a: int, b: int) -> int:
        """返回 [a, b] 区间内的均匀随机整数。"""
        return self._rng.randint(a, b)

    def choice(self, seq: Sequence[T]) -> T:
        """从序列中等概率随机选择一个元素。"""
        return self._rng.choice(seq)

    def shuffle(self, seq: List[Any]) -> None:
        """原地随机打乱序列。"""
        self._rng.shuffle(seq)

    def getstate(self) -> Any:
        return self._rng.getstate()

    def setstate(self, state: Any) -> None:
        self._rng.setstate(state)

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng.seed(seed)


_default_rng = UniformRNG()


# ============================================================================
# 二、逆变换采样 (Inverse Transform Sampling)
# ============================================================================

def inverse_transform_sample(
    inverse_cdf: Callable[[float], float],
    rng: Optional[UniformRNG] = None,
) -> float:
    """
    逆变换采样通用框架。

    原理: 若 U ~ Uniform(0,1), 且 F 是某分布的 CDF, 则 X = F^{-1}(U)
          服从该分布。前提: CDF F 必须可逆 (严格单调递增)。

    参数:
        inverse_cdf: 逆 CDF 函数 (分位数函数), 接受 [0,1) 的概率值, 返回对应的分位数值
        rng: 均匀随机数生成器 (可选)

    返回:
        服从目标分布的样本
    """
    rng = rng or _default_rng
    u = rng.random_open()
    return inverse_cdf(u)


def sample_exponential(lam: float = 1.0, rng: Optional[UniformRNG] = None) -> float:
    """
    指数分布采样 (使用逆变换法)。

    指数分布的 PDF: f(x) = λ * e^(-λx),  x >= 0
    指数分布的 CDF: F(x) = 1 - e^(-λx)
    逆 CDF:        F^{-1}(u) = -ln(1-u) / λ

    参数:
        lam: 率参数 λ (lambda), 必须 > 0
        rng: 均匀随机数生成器 (可选)

    返回:
        服从指数分布 Exp(λ) 的样本
    """
    if lam <= 0:
        raise ValueError("lambda 必须为正数")
    rng = rng or _default_rng
    u = rng.random_open()
    return -math.log(u) / lam


def sample_uniform(a: float, b: float, rng: Optional[UniformRNG] = None) -> float:
    """
    连续均匀分布采样 (逆变换法的最简情形)。

    参数:
        a: 下界
        b: 上界, 必须 b > a
    """
    if b <= a:
        raise ValueError("上界 b 必须大于下界 a")
    rng = rng or _default_rng
    return a + (b - a) * rng.random()


# ============================================================================
# 三、正态分布采样 (Box-Muller 变换)
# ============================================================================

def sample_normal_boxmuller(
    mu: float = 0.0,
    sigma: float = 1.0,
    rng: Optional[UniformRNG] = None,
) -> float:
    """
    正态分布采样 (Box-Muller 方法)。

    Box-Muller 变换原理:
    --------------------
    若 U1, U2 是独立的 Uniform(0,1) 随机变量, 令:
        R = sqrt(-2 * ln(U1))
        θ = 2π * U2
    则:
        Z1 = R * cos(θ)
        Z2 = R * sin(θ)
    是两个独立的标准正态分布 N(0,1) 样本。

    该方法每次调用生成两个独立的正态样本, 其中一个被缓存供下次使用,
    避免重复计算。

    参数:
        mu:    均值 μ
        sigma: 标准差 σ, 必须 > 0
        rng:   均匀随机数生成器 (可选)

    返回:
        服从正态分布 N(μ, σ²) 的样本
    """
    if sigma <= 0:
        raise ValueError("sigma 必须为正数")
    rng = rng or _default_rng

    if rng._normal_cache:
        z = rng._normal_cache.popleft()
        return mu + sigma * z

    u1 = rng.random_open()
    u2 = rng.random_open()

    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2

    z1 = r * math.cos(theta)
    z2 = r * math.sin(theta)

    rng._normal_cache.append(z2)
    return mu + sigma * z1


def sample_normal_pair_boxmuller(
    mu: float = 0.0,
    sigma: float = 1.0,
    rng: Optional[UniformRNG] = None,
) -> Tuple[float, float]:
    """
    Box-Muller 方法: 一次生成两个独立的正态分布样本。

    返回:
        (z1, z2): 两个独立的 N(μ, σ²) 样本
    """
    if sigma <= 0:
        raise ValueError("sigma 必须为正数")
    rng = rng or _default_rng

    u1 = rng.random_open()
    u2 = rng.random_open()

    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2

    z1 = r * math.cos(theta)
    z2 = r * math.sin(theta)

    return (mu + sigma * z1, mu + sigma * z2)


# ============================================================================
# 四、泊松分布采样
# ============================================================================

def sample_poisson(lam: float, rng: Optional[UniformRNG] = None) -> int:
    """
    泊松分布采样 (Knuth 算法, 适用于 λ 较小的情形)。

    原理:
    泊松过程中, 相邻事件的时间间隔服从指数分布 Exp(λ)。
    我们不断累加指数分布样本, 直到累加和超过 1, 累加的次数即为
    单位时间内发生的事件数, 服从泊松分布 Poisson(λ)。

    等价算法 (乘法形式, 避免对数运算):
        初始化 k = 0, p = 1
        循环:
            k += 1
            p *= U  (U 是新的均匀随机数)
            若 p < e^(-λ), 则返回 k-1

    时间复杂度: O(λ)

    参数:
        lam: 参数 λ, 必须 > 0 (均值和方差均为 λ)
        rng: 均匀随机数生成器 (可选)

    返回:
        服从泊松分布 Poisson(λ) 的非负整数样本
    """
    if lam <= 0:
        raise ValueError("lambda 必须为正数")
    rng = rng or _default_rng

    if lam < 30.0:
        return _sample_poisson_small(lam, rng)
    elif lam < 500.0:
        return _sample_poisson_medium(lam, rng)
    else:
        return _sample_poisson_huge(lam, rng)


def _sample_poisson_small(lam: float, rng: UniformRNG) -> int:
    """Knuth 乘法算法: 适用于 λ 较小 (< 30) 的情形。"""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random_open()
        if p < L:
            return k - 1


def _sample_poisson_medium(lam: float, rng: UniformRNG) -> int:
    """
    截断范围 CDF 递推法: 适用于 30 <= λ < 500。

    思想: 泊松分布 99.99999% 的概率集中在 [mode - 7σ, mode + 7σ] 范围内,
    因此只在这个区间内用递推累积, 步数 O(σ) = O(sqrt(λ))。
    递推关系: P(k+1) = P(k) * λ / (k+1)

    若极小概率 u 落在截断范围左侧, 则退化为从 0 开始的慢速累积 (几乎不触发)。
    数值绝对稳定, 精度精确, 不会死循环。
    """
    u = rng.random_open()
    sigma = math.sqrt(lam)
    mode = int(math.floor(lam))

    tail_sigma = 7
    start_k = max(0, mode - int(tail_sigma * sigma) - 1)

    log_p = start_k * math.log(lam) - lam - math.lgamma(start_k + 1)
    p = math.exp(log_p)
    cdf = 0.0
    k = start_k

    right_bound = mode + int(tail_sigma * sigma) + 10

    while k <= right_bound:
        cdf += p
        if u < cdf:
            return k
        p *= lam / (k + 1)
        k += 1

    return _sample_poisson_from_zero(lam, u)


def _sample_poisson_from_zero(lam: float, u: float) -> int:
    """从 k=0 开始正向累积 CDF 的慢速保底路径。"""
    p = math.exp(-lam)
    cdf = 0.0
    k = 0
    while True:
        cdf += p
        if u < cdf:
            return k
        p *= lam / (k + 1)
        k += 1


def _sample_poisson_huge(lam: float, rng: UniformRNG) -> int:
    """
    正态近似 + 边界修正: 适用于 λ >= 500。

    当 λ → ∞, 由中心极限定理:
        (X - λ) / √λ  →  N(0, 1)  (依分布收敛)

    λ ≥ 500 时偏度 ≈ 1/√λ ≈ 0.045, 正态近似误差极小,
    且速度恒定 O(1), 不会出现拒绝采样的数值不稳定。
    """
    sigma = math.sqrt(lam)
    z = sample_normal_boxmuller(0.0, 1.0, rng)
    n = int(math.floor(lam + sigma * z + 0.5))
    return 0 if n < 0 else n


# ============================================================================
# 五、离散分布别名方法 (Alias Method / Walker's Alias Method)
# ============================================================================

class AliasSampler:
    """
    Walker 别名方法: 任意离散分布的 O(1) 采样。

    预处理: O(n) 时间构建别名表
    采样:   O(1) 时间 (两次查表 + 一次均匀随机数比较)

    原理 (直观理解):
    ----------------
    给定 n 个结果, 概率分别为 p_1, p_2, ..., p_n。

    第 1 步: 将每个概率乘以 n, 得到 q_i = n * p_i。
             每个 q_i 表示该结果占据的"面积比例"。

    第 2 步: 将结果分为两组:
             - 欠载 (underfull): q_i < 1
             - 过载 (overfull):  q_i >= 1

    第 3 步: 每次从两组各取一个, 用过载的部分去"填补"欠载的部分,
             直到所有槽位都恰好填满 (面积 = 1)。

    构建结果: 两个长度为 n 的数组
        - prob[i]: 第 i 个槽位选择"本尊"的概率
        - alias[i]: 第 i 个槽位选择"别名"时对应的索引

    采样过程:
        1. 均匀随机选择一个槽位 i (0 <= i < n)
        2. 生成一个 [0,1) 的均匀随机数 u
        3. 若 u < prob[i], 返回 i; 否则返回 alias[i]
    """

    def __init__(self, probabilities: Sequence[float], rng: Optional[UniformRNG] = None) -> None:
        """
        初始化别名采样器。

        参数:
            probabilities: 离散概率分布或非负权重, 长度为 n;
                           支持未归一化权重 (会自动归一化), 但不允许负值
            rng: 均匀随机数生成器 (可选)

        异常:
            ValueError: 空列表、含有负值、所有权重为 0
        """
        probs = list(probabilities)
        n = len(probs)
        if n == 0:
            raise ValueError("概率分布不能为空列表")

        for i, p in enumerate(probs):
            if p < 0:
                raise ValueError(
                    f"概率/权重不能为负值: 索引 {i} 的值为 {p!r}"
                )

        total = sum(probs)
        if total == 0:
            raise ValueError("所有概率/权重均为 0, 无法确定有效分布")
        probs = [p / total for p in probs]

        self.n = n
        self._rng = rng or _default_rng
        self.prob: List[float] = [0.0] * n
        self.alias: List[int] = [0] * n

        self._build_alias_table(probs)

    def _build_alias_table(self, probs: List[float]) -> None:
        """Vose 算法: O(n) 时间构建别名表。"""
        n = self.n
        scaled = [p * n for p in probs]

        underfull: deque = deque()
        overfull: deque = deque()

        for i, q in enumerate(scaled):
            if q < 1.0:
                underfull.append(i)
            elif q >= 1.0:
                overfull.append(i)

        while underfull and overfull:
            under = underfull.popleft()
            over = overfull.popleft()

            self.prob[under] = scaled[under]
            self.alias[under] = over

            scaled[over] = (scaled[over] + scaled[under]) - 1.0

            if scaled[over] < 1.0:
                underfull.append(over)
            elif scaled[over] >= 1.0:
                overfull.append(over)

        while underfull:
            i = underfull.popleft()
            self.prob[i] = 1.0
            self.alias[i] = i

        while overfull:
            i = overfull.popleft()
            self.prob[i] = 1.0
            self.alias[i] = i

    def sample(self) -> int:
        """
        采样一个离散结果。

        返回:
            0 到 n-1 之间的整数索引
        """
        i = self._rng.randint(0, self.n - 1)
        u = self._rng.random()
        if u < self.prob[i]:
            return i
        else:
            return self.alias[i]

    def sample_index(self) -> int:
        """sample() 的别名。"""
        return self.sample()


def sample_discrete_alias(
    probabilities: Sequence[float],
    rng: Optional[UniformRNG] = None,
) -> int:
    """
    使用别名方法从离散分布中采样。

    注意: 如果需要多次采样, 建议直接创建 AliasSampler 实例复用,
          以避免重复构建别名表。
    """
    sampler = AliasSampler(probabilities, rng)
    return sampler.sample()


# ============================================================================
# 六、加权随机抽样
# ============================================================================

def sample_weighted(
    items: Sequence[T],
    weights: Sequence[float],
    k: int = 1,
    replace: bool = True,
    rng: Optional[UniformRNG] = None,
) -> List[T]:
    """
    加权随机抽样。

    参数:
        items:    候选物品序列
        weights:  对应的权重序列 (非负, 不需归一化)
        k:        抽样数量
        replace:  是否有放回抽样 (True=有放回, False=无放回)
        rng:      均匀随机数生成器 (可选)

    返回:
        长度为 k 的抽样结果列表
    """
    if len(items) != len(weights):
        raise ValueError("items 和 weights 长度必须相同")
    if any(w < 0 for w in weights):
        raise ValueError("权重不能为负")
    if k <= 0:
        raise ValueError("抽样数量 k 必须为正整数")

    rng = rng or _default_rng

    if replace:
        return _sample_weighted_with_replacement(items, weights, k, rng)
    else:
        return _sample_weighted_without_replacement(items, weights, k, rng)


def _sample_weighted_with_replacement(
    items: Sequence[T],
    weights: Sequence[float],
    k: int,
    rng: UniformRNG,
) -> List[T]:
    """有放回加权抽样: 使用别名方法实现 O(1) 每次采样。"""
    sampler = AliasSampler(weights, rng)
    return [items[sampler.sample()] for _ in range(k)]


def _sample_weighted_without_replacement(
    items: Sequence[T],
    weights: Sequence[float],
    k: int,
    rng: UniformRNG,
) -> List[T]:
    """
    无放回加权抽样: 使用 Efraimidis-Spirakis 算法 (O(n log n))。

    原理: 对每个物品 i, 计算 u_i = U^(1/w_i), 其中 U ~ Uniform(0,1)。
          选择 k 个 u_i 最大的物品即为所需样本。

    注意: 权重为 0 的物品永远不会被抽中。
    """
    n = len(items)
    if k > n:
        raise ValueError(
            f"无放回抽样时 k(={k}) 不能超过物品总数(={n})"
        )

    positive_weight_indices = [i for i in range(n) if weights[i] > 0]
    num_positive = len(positive_weight_indices)

    if num_positive < k:
        zero_weight_items = [
            (items[i], weights[i])
            for i in range(n)
            if weights[i] <= 0
        ]
        raise ValueError(
            f"正权重物品数量不足: 需要抽 {k} 个, "
            f"但权重 > 0 的物品只有 {num_positive} 个 "
            f"(零/负权重物品共 {n - num_positive} 个: {zero_weight_items!r})"
        )

    keys: List[Tuple[float, int]] = []
    for i in range(n):
        w = weights[i]
        if w <= 0:
            keys.append((float("-inf"), i))
        else:
            u = rng.random_open()
            key = math.log(u) / w
            keys.append((key, i))

    keys.sort(reverse=True)
    return [items[idx] for (_, idx) in keys[:k]]


# ============================================================================
# 七、辅助工具函数
# ============================================================================

def sample_multinomial(
    n_trials: int,
    probabilities: Sequence[float],
    rng: Optional[UniformRNG] = None,
) -> List[int]:
    """
    多项分布采样。

    参数:
        n_trials:      试验次数
        probabilities: 每个类别出现的概率 (和为 1)

    返回:
        每个类别出现的次数 (长度与 probabilities 相同)
    """
    rng = rng or _default_rng
    sampler = AliasSampler(probabilities, rng)
    counts = [0] * len(probabilities)
    for _ in range(n_trials):
        counts[sampler.sample()] += 1
    return counts


def sample_bernoulli(p: float, rng: Optional[UniformRNG] = None) -> int:
    """伯努利分布采样: 以概率 p 返回 1, 以概率 1-p 返回 0。"""
    if not (0.0 <= p <= 1.0):
        raise ValueError("p 必须在 [0, 1] 区间内")
    rng = rng or _default_rng
    return 1 if rng.random() < p else 0


def sample_gamma(
    shape: float,
    rate: float = 1.0,
    rng: Optional[UniformRNG] = None,
) -> float:
    """
    Gamma 分布采样 (Marsaglia-Tsang 方法, 2000)。

    Gamma(shape=α, rate=β) 的 PDF:
        f(x) = (β^α / Γ(α)) * x^(α-1) * e^(-βx),  x > 0

    均值 = α/β,  方差 = α/β²

    参数:
        shape: 形状参数 α > 0
        rate:  率参数 β > 0 (默认 1.0)
        rng:   均匀随机数生成器 (可选)

    返回:
        服从 Gamma(α, β) 的样本
    """
    if shape <= 0:
        raise ValueError("形状参数 shape 必须为正数")
    if rate <= 0:
        raise ValueError("率参数 rate 必须为正数")
    rng = rng or _default_rng
    return _sample_gamma(shape, rate, rng)


def sample_beta(alpha: float, beta: float, rng: Optional[UniformRNG] = None) -> float:
    """
    Beta 分布采样 (基于两个 Gamma 分布)。

    Beta(α, β) 的 PDF:
        f(x) = x^(α-1) * (1-x)^(β-1) / B(α, β),  0 < x < 1

    均值 = α/(α+β),  方差 = αβ / [(α+β)²(α+β+1)]

    参数:
        alpha: 形状参数 α > 0
        beta:  形状参数 β > 0
        rng:   均匀随机数生成器 (可选)

    返回:
        服从 Beta(α, β) 的样本 (0 < x < 1)
    """
    if alpha <= 0:
        raise ValueError(f"形状参数 alpha 必须为正数, 得到 {alpha!r}")
    if beta <= 0:
        raise ValueError(f"形状参数 beta 必须为正数, 得到 {beta!r}")
    rng = rng or _default_rng
    x = _sample_gamma(alpha, 1.0, rng)
    y = _sample_gamma(beta, 1.0, rng)
    return x / (x + y)


def sample_binomial(
    n: int,
    p: float,
    rng: Optional[UniformRNG] = None,
) -> int:
    """
    二项分布采样。

    Binomial(n, p) 描述 n 次独立伯努利试验中成功的次数,
    PMF: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)

    均值 = n*p,  方差 = n*p*(1-p)

    参数:
        n:   试验次数, 正整数
        p:   单次试验成功概率, 0 <= p <= 1
        rng: 均匀随机数生成器 (可选)

    返回:
        成功次数 k ∈ {0, 1, ..., n}
    """
    if n <= 0:
        raise ValueError("试验次数 n 必须为正整数")
    if not (0.0 <= p <= 1.0):
        raise ValueError("概率 p 必须在 [0, 1] 区间内")
    rng = rng or _default_rng

    if p == 0.0:
        return 0
    if p == 1.0:
        return n

    if n * p < 10:
        return sum(1 for _ in range(n) if rng.random() < p)
    else:
        mu = n * p
        sigma = math.sqrt(n * p * (1.0 - p))
        k = int(math.floor(sample_normal_boxmuller(mu, sigma, rng) + 0.5))
        if k < 0:
            return 0
        if k > n:
            return n
        return k


def sample_geometric(
    p: float,
    rng: Optional[UniformRNG] = None,
) -> int:
    """
    几何分布采样 (逆变换法)。

    几何分布描述首次成功前的失败次数, PMF:
        P(X=k) = (1-p)^k * p,   k = 0, 1, 2, ...

    均值 = (1-p)/p,  方差 = (1-p)/p²

    逆变换公式:
        k = ⌊ ln(1-U) / ln(1-p) ⌋

    参数:
        p:   单次试验成功概率, 0 < p <= 1
        rng: 均匀随机数生成器 (可选)

    返回:
        首次成功前的失败次数 k >= 0
    """
    if not (0.0 < p <= 1.0):
        raise ValueError("概率 p 必须在 (0, 1] 区间内")
    if p == 1.0:
        return 0
    rng = rng or _default_rng
    u = rng.random_open()
    return int(math.floor(math.log(1.0 - u) / math.log(1.0 - p)))


def _sample_gamma(shape: float, rate: float, rng: UniformRNG) -> float:
    """Gamma 分布采样 (Marsaglia-Tsang 方法)。"""
    if shape < 1.0:
        return _sample_gamma(shape + 1.0, rate, rng) * (rng.random_open() ** (1.0 / shape))

    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)

    while True:
        while True:
            z = sample_normal_boxmuller(0.0, 1.0, rng)
            if z > -1.0 / c:
                break
        v = (1.0 + c * z) ** 3
        u = rng.random_open()

        if u < 1.0 - 0.0331 * (z * z) * (z * z):
            return d * v / rate
        if math.log(u) < 0.5 * z * z + d * (1.0 - v + math.log(v)):
            return d * v / rate


# ============================================================================
# 八、批量采样便利函数
# ============================================================================

def batch_sample(
    sampler_fn: Callable[..., T],
    n_samples: int,
    *args,
    **kwargs,
) -> List[T]:
    """
    批量采样工具: 调用采样函数 n 次并返回结果列表。

    参数:
        sampler_fn: 采样函数 (如 sample_normal_boxmuller)
        n_samples:  样本数量
        *args:      传递给采样函数的位置参数
        **kwargs:   传递给采样函数的关键字参数

    返回:
        长度为 n_samples 的样本列表

    示例:
        samples = batch_sample(sample_normal_boxmuller, 10000, mu=0, sigma=1)
    """
    return [sampler_fn(*args, **kwargs) for _ in range(n_samples)]


# ============================================================================
# 九、统计摘要与导出工具
# ============================================================================

def compute_statistics(
    samples: Sequence[float],
) -> Dict[str, Union[int, float, List[float]]]:
    """
    计算样本的基本统计摘要。

    参数:
        samples: 数值样本序列

    返回:
        包含以下键的字典:
        - count: 样本量
        - mean: 均值
        - variance: 方差 (除以 n, 非无偏估计)
        - std: 标准差
        - min: 最小值
        - max: 最大值
        - median: 中位数 (50% 分位数)
        - q25: 25% 分位数
        - q75: 75% 分位数
        - q05: 5% 分位数
        - q95: 95% 分位数
        - quantiles: [5%, 25%, 50%, 75%, 95%] 分位数列表
    """
    n = len(samples)
    if n == 0:
        raise ValueError("样本序列不能为空")

    s = sorted(samples)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    std = math.sqrt(variance)

    def quantile(data_sorted: List[float], q: float) -> float:
        """线性插值分位数。"""
        m = len(data_sorted)
        if m == 1:
            return data_sorted[0]
        pos = q * (m - 1)
        idx = int(math.floor(pos))
        frac = pos - idx
        if idx + 1 >= m:
            return data_sorted[-1]
        return data_sorted[idx] + frac * (data_sorted[idx + 1] - data_sorted[idx])

    q05 = quantile(s, 0.05)
    q25 = quantile(s, 0.25)
    q50 = quantile(s, 0.50)
    q75 = quantile(s, 0.75)
    q95 = quantile(s, 0.95)

    return {
        "count": n,
        "mean": mean,
        "variance": variance,
        "std": std,
        "min": s[0],
        "max": s[-1],
        "median": q50,
        "q25": q25,
        "q75": q75,
        "q05": q05,
        "q95": q95,
        "quantiles": [q05, q25, q50, q75, q95],
    }


def format_statistics_report(
    stats: Dict[str, Union[int, float, List[float]]],
    title: str = "统计摘要",
    theoretical_mean: Optional[float] = None,
    theoretical_var: Optional[float] = None,
) -> str:
    """
    格式化统计摘要为可读字符串。

    参数:
        stats: compute_statistics() 返回的字典
        title: 报告标题
        theoretical_mean: 理论均值 (可选, 用于对比)
        theoretical_var: 理论方差 (可选, 用于对比)

    返回:
        格式化的多行字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {title}")
    lines.append("=" * 60)
    lines.append(f"  样本量 (N)    : {stats['count']}")
    lines.append("-" * 60)

    lines.append(f"  均值          : {stats['mean']:.6f}")
    if theoretical_mean is not None:
        abs_dev = stats['mean'] - theoretical_mean
        if abs(theoretical_mean) < 1e-10:
            lines.append(
                f"    理论均值    : {theoretical_mean:.6f}  "
                f"(绝对偏差 {abs_dev:+.6f}, 百分比不适用)"
            )
        else:
            pct_dev = abs_dev / theoretical_mean * 100
            lines.append(
                f"    理论均值    : {theoretical_mean:.6f}  "
                f"(偏差 {pct_dev:+.3f}%)"
            )

    lines.append(f"  方差          : {stats['variance']:.6f}")
    if theoretical_var is not None:
        abs_dev = stats['variance'] - theoretical_var
        if abs(theoretical_var) < 1e-10:
            lines.append(
                f"    理论方差    : {theoretical_var:.6f}  "
                f"(绝对偏差 {abs_dev:+.6f}, 百分比不适用)"
            )
        else:
            pct_dev = abs_dev / theoretical_var * 100
            lines.append(
                f"    理论方差    : {theoretical_var:.6f}  "
                f"(偏差 {pct_dev:+.3f}%)"
            )

    lines.append(f"  标准差        : {stats['std']:.6f}")
    lines.append(f"  最小值        : {stats['min']:.6f}")
    lines.append(f"  最大值        : {stats['max']:.6f}")
    lines.append("-" * 60)
    lines.append("  分位数:")
    lines.append(f"     5%  = {stats['q05']:.6f}")
    lines.append(f"    25%  = {stats['q25']:.6f}")
    lines.append(f"    50%  = {stats['median']:.6f}   (中位数)")
    lines.append(f"    75%  = {stats['q75']:.6f}")
    lines.append(f"    95%  = {stats['q95']:.6f}")
    lines.append("=" * 60)

    return "\n".join(lines)


def export_samples_csv(
    samples: Sequence[float],
    filepath: str,
    header: str = "value",
    include_index: bool = False,
) -> str:
    """
    将样本导出为 CSV 文件。

    参数:
        samples: 数值样本序列
        filepath: 输出 CSV 文件路径
        header: 列标题 (默认 "value")
        include_index: 是否包含序号列 (默认 False)

    返回:
        实际写入的文件绝对路径
    """
    import os
    abs_path = os.path.abspath(filepath)

    with open(abs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if include_index:
            writer.writerow(["index", header])
            for i, val in enumerate(samples):
                writer.writerow([i, val])
        else:
            writer.writerow([header])
            for val in samples:
                writer.writerow([val])

    return abs_path


def generate_report_json(
    distribution_name: str,
    parameters: Dict[str, Any],
    stats: Dict[str, Union[int, float, List[float]]],
    seed: Optional[int] = None,
    theoretical_mean: Optional[float] = None,
    theoretical_var: Optional[float] = None,
) -> Dict[str, Any]:
    """
    生成一份包含采样元数据与统计结果的结构化报告 (字典形式)。

    参数:
        distribution_name: 分布名称 (如 "normal", "poisson")
        parameters: 分布参数字典 (如 {"mu": 0, "sigma": 1})
        stats: compute_statistics() 返回的摘要
        seed: 使用的随机种子 (可选)
        theoretical_mean: 理论均值 (可选)
        theoretical_var: 理论方差 (可选)

    返回:
        结构化报告字典, 可直接序列化为 JSON
    """
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "distribution": distribution_name,
        "parameters": parameters,
        "seed": seed,
        "sample_size": int(stats["count"]),
        "statistics": {
            "mean": float(stats["mean"]),
            "variance": float(stats["variance"]),
            "std": float(stats["std"]),
            "min": float(stats["min"]),
            "max": float(stats["max"]),
            "median": float(stats["median"]),
            "quantiles": {
                "5%": float(stats["q05"]),
                "25%": float(stats["q25"]),
                "50%": float(stats["median"]),
                "75%": float(stats["q75"]),
                "95%": float(stats["q95"]),
            },
        },
    }

    if theoretical_mean is not None:
        report["theoretical_mean"] = float(theoretical_mean)
        abs_dev = float(stats["mean"] - theoretical_mean)
        report["mean_deviation_abs"] = abs_dev
        if abs(theoretical_mean) < 1e-10:
            report["mean_deviation_pct"] = None
            report["mean_deviation_note"] = "理论均值为 0, 百分比偏差不适用"
        else:
            report["mean_deviation_pct"] = float(
                abs_dev / theoretical_mean * 100
            )

    if theoretical_var is not None:
        report["theoretical_variance"] = float(theoretical_var)
        abs_dev_var = float(stats["variance"] - theoretical_var)
        report["variance_deviation_abs"] = abs_dev_var
        if abs(theoretical_var) < 1e-10:
            report["variance_deviation_pct"] = None
            report["variance_deviation_note"] = "理论方差为 0, 百分比偏差不适用"
        else:
            report["variance_deviation_pct"] = float(
                abs_dev_var / theoretical_var * 100
            )

    return report


def save_report_json(report: Dict[str, Any], filepath: str) -> str:
    """将结构化报告保存为 JSON 文件。"""
    import os
    abs_path = os.path.abspath(filepath)
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return abs_path


# ============================================================================
# 十、直方图分箱统计
# ============================================================================

def compute_histogram(
    samples: Sequence[float],
    bins: int = 20,
    value_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    计算样本的直方图分箱统计（等距分箱）。

    参数:
        samples:     数值样本序列
        bins:        分箱数量 (默认 20)
        value_range: 分箱范围 (min, max), None 则使用数据的 min/max

    返回:
        包含以下键的字典:
        - bin_edges: 分箱边界列表 (长度 = bins + 1)
        - bin_centers: 每个箱的中心点列表 (长度 = bins)
        - bin_width: 分箱宽度
        - counts: 每个箱内的样本数 (长度 = bins)
        - frequencies: 每个箱内的样本频率 (长度 = bins)
        - range: (min, max) 实际使用的范围
    """
    n = len(samples)
    if n == 0:
        raise ValueError("样本序列不能为空")
    if bins <= 0:
        raise ValueError(f"分箱数量必须为正整数, 得到 {bins}")

    data_min, data_max = min(samples), max(samples)
    if value_range is not None:
        hist_min, hist_max = value_range
        if hist_min >= hist_max:
            raise ValueError(
                f"分箱范围无效: {value_range!r}, 需要 (min, max) 且 min < max"
            )
    else:
        hist_min, hist_max = data_min, data_max

    if hist_min == hist_max:
        hist_max = hist_min + 1.0

    bin_width = (hist_max - hist_min) / bins

    counts = [0] * bins
    for x in samples:
        if x < hist_min or x > hist_max:
            continue
        if x == hist_max:
            idx = bins - 1
        else:
            idx = int((x - hist_min) / bin_width)
            if idx >= bins:
                idx = bins - 1
            if idx < 0:
                idx = 0
        counts[idx] += 1

    bin_edges = [hist_min + i * bin_width for i in range(bins + 1)]
    bin_centers = [hist_min + (i + 0.5) * bin_width for i in range(bins)]
    frequencies = [c / n for c in counts]

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_width": bin_width,
        "counts": counts,
        "frequencies": frequencies,
        "range": (hist_min, hist_max),
    }


def export_histogram_csv(
    hist: Dict[str, Any],
    filepath: str,
    include_edges: bool = True,
) -> str:
    """
    将直方图统计导出为 CSV 文件, 方便后续在 Excel / matplotlib 中画图。

    CSV 列:
        bin_index, bin_center, bin_width, count, frequency
        [可选] bin_left, bin_right

    参数:
        hist:         compute_histogram() 返回的直方图数据
        filepath:     输出 CSV 路径
        include_edges: 是否包含 bin_left 和 bin_right 列 (默认 True)

    返回:
        实际写入的绝对路径
    """
    import os
    abs_path = os.path.abspath(filepath)

    counts = hist["counts"]
    bins = len(counts)
    centers = hist["bin_centers"]
    edges = hist["bin_edges"]
    width = hist["bin_width"]
    freqs = hist["frequencies"]

    with open(abs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["bin_index", "bin_center", "bin_width", "count", "frequency"]
        if include_edges:
            header += ["bin_left", "bin_right"]
        writer.writerow(header)

        for i in range(bins):
            row = [i, f"{centers[i]:.6f}", f"{width:.6f}", counts[i], f"{freqs[i]:.6f}"]
            if include_edges:
                row += [f"{edges[i]:.6f}", f"{edges[i + 1]:.6f}"]
            writer.writerow(row)

    return abs_path


# ============================================================================
# 十一、批量实验配置与汇总报告
# ============================================================================

def load_batch_config(filepath: str) -> Dict[str, Any]:
    """
    从 JSON 或 YAML 文件加载批量实验配置。

    根据扩展名自动识别格式 (.json / .yaml / .yml)。

    支持公共默认参数 defaults 字段，单组实验只写差异项即可。

    配置文件格式示例 (JSON):
    {
      "global_seed": 42,
      "output_dir": "./batch_results",
      "export_samples": true,
      "export_reports": true,
      "export_histograms": true,
      "histogram_bins": 30,
      "defaults": {
        "num_samples": 30000,
        "seed": 42
      },
      "experiments": [
        {"name": "normal_0_1",   "distribution": "normal", "params": {"mu": 0, "sigma": 1}},
        {"name": "normal_5_2",   "distribution": "normal", "params": {"mu": 5, "sigma": 2}},
        {"name": "poisson_10",     "distribution": "poisson", "params": {"lam": 10}, "num_samples": 20000},
        {"name": "beta_2_5",     "distribution": "beta",   "params": {"alpha": 2, "beta": 5}}
      ]
    }

    配置文件格式示例 (YAML):
    global_seed: 42
    output_dir: ./batch_results
    histogram_bins: 30
    defaults:
      num_samples: 30000
    experiments:
      - name: normal_0_1
        distribution: normal
        params:
          mu: 0
          sigma: 1
      - name: normal_5_2
        distribution: normal
        params:
          mu: 5
          sigma: 2

    每个实验的配置项 (可继承 defaults:
      - name:         实验名称 (可选, 默认为 distribution_N)
      - distribution: 分布名称 (必填)
      - params:       分布参数字典 (必填)
      - num_samples: 样本数量 (默认 10000)
      - seed:         单独随机种子 (可选, 覆盖 global_seed)

    参数:
        filepath: JSON 或 YAML 配置文件路径

    返回:
        解析并合并默认值后的配置字典
    """
    import os

    ext = os.path.splitext(filepath)[1].lower()

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if ext in (".yaml", ".yml"):
        try:
            import yaml
            config = yaml.safe_load(content)
        except ImportError:
            raise ImportError(
                "加载 YAML 配置需要 PyYAML, 请运行: pip install pyyaml"
            )
    elif ext == ".json":
        config = json.loads(content)
    else:
        try:
            config = json.loads(content)
        except Exception:
            try:
                import yaml
                config = yaml.safe_load(content)
            except Exception as e2:
                raise ValueError(
                    f"无法解析配置文件 (尝试 JSON 和 YAML 均失败。请检查文件格式。"
                f"YAML 错误: {e2}"
                )

    if not isinstance(config, dict):
        raise ValueError("配置文件根节点必须是字典/对象")

    if "experiments" not in config or not isinstance(config["experiments"], list):
        raise ValueError("配置文件必须包含 'experiments' 列表")

    # 全局默认值
    defaults = config.get("defaults", {}) or {}
    default_num_samples = defaults.get("num_samples", 10000)
    default_seed = defaults.get("seed")
    default_distribution = defaults.get("distribution")
    default_params = defaults.get("params", {}) or {}

    merged_experiments = []
    for i, exp in enumerate(config["experiments"]):
        if not isinstance(exp, dict):
            raise ValueError(f"第 {i} 个实验必须是字典")

        merged = {}
        merged["distribution"] = exp.get("distribution", default_distribution)
        exp_params = exp.get("params", {}) or {}
        merged["params"] = {**default_params, **exp_params}
        merged["num_samples"] = exp.get("num_samples", default_num_samples)

        if "seed" in exp:
            merged["seed"] = exp["seed"]
        elif default_seed is not None:
            merged["seed"] = default_seed

        if "name" in exp:
            merged["name"] = exp["name"]

        if not merged.get("distribution"):
            raise ValueError(
                f"第 {i} 个实验缺少 'distribution' 字段 (可在 defaults 或该实验中指定)"
            )
        if not merged.get("params"):
            raise ValueError(
                f"第 {i} 个实验缺少 'params' 字段 (可在 defaults 或该实验中指定)"
            )

        merged_experiments.append(merged)

    config["experiments"] = merged_experiments

    config.setdefault("global_seed", None)
    config.setdefault("output_dir", "./batch_results")
    config.setdefault("export_samples", True)
    config.setdefault("export_reports", True)
    config.setdefault("export_histograms", True)
    config.setdefault("histogram_bins", 20)

    return config


_BATCH_DISTRIBUTION_SAMPLERS: Dict[str, Tuple[Callable, Callable]] = {
    "uniform":     (sample_uniform,     lambda p: ((p["a"] + p["b"]) / 2.0, (p["b"] - p["a"]) ** 2 / 12.0)),
    "exponential": (sample_exponential, lambda p: (1.0 / p["lam"], 1.0 / (p["lam"] ** 2))),
    "normal":      (sample_normal_boxmuller, lambda p: (p["mu"], p["sigma"] ** 2)),
    "poisson":     (sample_poisson,     lambda p: (p["lam"], p["lam"])),
    "gamma":       (sample_gamma,       lambda p: (p["shape"] / p["rate"], p["shape"] / (p["rate"] ** 2))),
    "beta":        (sample_beta,        lambda p: (p["alpha"] / (p["alpha"] + p["beta"]),
                                                  (p["alpha"] * p["beta"]) / ((p["alpha"] + p["beta"]) ** 2 * (p["alpha"] + p["beta"] + 1)))),
    "binomial":    (sample_binomial,    lambda p: (p["n"] * p["p"], p["n"] * p["p"] * (1.0 - p["p"]))),
    "geometric":   (sample_geometric,   lambda p: ((1.0 - p["p"]) / p["p"], (1.0 - p["p"]) / (p["p"] ** 2))),
    "bernoulli":   (sample_bernoulli,   lambda p: (p["p"], p["p"] * (1.0 - p["p"]))),
}


def run_batch_experiments(
    config: Dict[str, Any],
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    运行批量实验, 生成样本、报告和直方图, 并返回汇总结果。

    参数:
        config: load_batch_config() 返回的配置字典
        quiet:  是否不打印进度信息

    返回:
        汇总结果字典, 包含:
        - summary: 各实验统计摘要的列表 (用于生成汇总表格)
        - results: 各实验的完整结果 (样本、统计、直方图等)
        - output_dir: 输出目录
    """
    import os

    output_dir = os.path.abspath(config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    global_seed = config.get("global_seed")
    export_samples = config.get("export_samples", True)
    export_reports = config.get("export_reports", True)
    export_histograms = config.get("export_histograms", True)
    histogram_bins = config.get("histogram_bins", 20)

    summary_rows: List[Dict[str, Any]] = []
    full_results: List[Dict[str, Any]] = []

    for exp_idx, exp in enumerate(config["experiments"]):
        dist_name = exp["distribution"]
        params = exp["params"]
        num_samples = exp.get("num_samples", 10000)
        seed = exp.get("seed", global_seed)
        exp_name = exp.get("name", f"{dist_name}_{exp_idx}")

        if not quiet:
            print(f"[{exp_idx + 1}/{len(config['experiments'])}] 运行 {exp_name} "
                  f"({dist_name}, N={num_samples})...", flush=True)

        if dist_name not in _BATCH_DISTRIBUTION_SAMPLERS:
            raise ValueError(f"不支持的分布: {dist_name!r}, 可用: {list(_BATCH_DISTRIBUTION_SAMPLERS.keys())}")

        sampler, theory_fn = _BATCH_DISTRIBUTION_SAMPLERS[dist_name]

        # 过滤参数: 只保留该分布 sampler 函数实际接受的参数,
        # 这样 defaults 中放的通用参数不会污染不相关的分布
        import inspect
        try:
            sig = inspect.signature(sampler)
            accepted = set(sig.parameters.keys()) - {"rng"}
            filtered_params = {k: v for k, v in params.items() if k in accepted}
        except (ValueError, TypeError):
            filtered_params = params

        rng = UniformRNG(seed=seed)
        samples = batch_sample(sampler, num_samples, rng=rng, **filtered_params)

        stats = compute_statistics(samples)
        theory_mean, theory_var = theory_fn(filtered_params)

        hist = None
        if export_histograms:
            hist = compute_histogram(samples, bins=histogram_bins)

        exp_result: Dict[str, Any] = {
            "name": exp_name,
            "distribution": dist_name,
            "params": filtered_params,
            "num_samples": num_samples,
            "seed": seed,
            "samples": samples,
            "stats": stats,
            "theoretical_mean": theory_mean,
            "theoretical_var": theory_var,
            "histogram": hist,
        }

        sample_path = None
        if export_samples:
            sample_path = os.path.join(output_dir, f"{exp_name}_samples.csv")
            export_samples_csv(samples, sample_path)
            exp_result["sample_csv_path"] = sample_path
            if not quiet:
                print(f"  ✓ 样本 CSV -> {sample_path}")

        report_path = None
        if export_reports:
            report = generate_report_json(
                dist_name, filtered_params, stats,
                seed=seed,
                theoretical_mean=theory_mean,
                theoretical_var=theory_var,
            )
            report_path = os.path.join(output_dir, f"{exp_name}_report.json")
            save_report_json(report, report_path)
            exp_result["report_json_path"] = report_path
            if not quiet:
                print(f"  ✓ 报告 JSON -> {report_path}")

        hist_path = None
        if export_histograms and hist is not None:
            hist_path = os.path.join(output_dir, f"{exp_name}_histogram.csv")
            export_histogram_csv(hist, hist_path)
            exp_result["histogram_csv_path"] = hist_path
            if not quiet:
                print(f"  ✓ 直方图 CSV -> {hist_path}")

        mean_dev_abs = stats["mean"] - theory_mean
        if abs(theory_mean) < 1e-10:
            mean_dev_pct = None
            mean_dev_note = "百分比不适用"
        else:
            mean_dev_pct = mean_dev_abs / theory_mean * 100
            mean_dev_note = ""

        var_dev_abs = stats["variance"] - theory_var
        if abs(theory_var) < 1e-10:
            var_dev_pct = None
            var_dev_note = "百分比不适用"
        else:
            var_dev_pct = var_dev_abs / theory_var * 100
            var_dev_note = ""

        summary_rows.append({
            "exp_name": exp_name,
            "distribution": dist_name,
            "params": filtered_params,
            "num_samples": num_samples,
            "seed": seed,
            "mean": float(stats["mean"]),
            "theoretical_mean": float(theory_mean),
            "mean_deviation_abs": float(mean_dev_abs),
            "mean_deviation_pct": float(mean_dev_pct) if mean_dev_pct is not None else None,
            "mean_deviation_note": mean_dev_note,
            "variance": float(stats["variance"]),
            "theoretical_variance": float(theory_var),
            "variance_deviation_abs": float(var_dev_abs),
            "variance_deviation_pct": float(var_dev_pct) if var_dev_pct is not None else None,
            "variance_deviation_note": var_dev_note,
            "std": float(stats["std"]),
            "min": float(stats["min"]),
            "max": float(stats["max"]),
            "q05": float(stats["q05"]),
            "q25": float(stats["q25"]),
            "median": float(stats["median"]),
            "q75": float(stats["q75"]),
            "q95": float(stats["q95"]),
            "sample_csv_path": sample_path,
            "report_json_path": report_path,
            "histogram_csv_path": hist_path,
        })

        full_results.append(exp_result)
        if not quiet:
            print()

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir,
        "num_experiments": len(summary_rows),
        "experiments": summary_rows,
    }

    return {
        "summary": summary,
        "results": full_results,
        "output_dir": output_dir,
    }


def format_batch_summary_table(summary: Dict[str, Any]) -> str:
    """
    将批量实验汇总结果格式化为终端可读的表格。

    参数:
        summary: run_batch_experiments() 返回的汇总字典

    返回:
        格式化的多行字符串表格
    """
    exps = summary["experiments"]
    if not exps:
        return "无实验数据"

    name_len = max(max(len(e["exp_name"]) for e in exps), 10)
    dist_len = max(max(len(e["distribution"]) for e in exps), 8)

    lines = []
    lines.append("=" * (name_len + dist_len + 130))
    header = (
        f"{'实验名称':<{name_len}}  "
        f"{'分布':<{dist_len}}  "
        f"{'N':>8}  "
        f"{'均值':>10}  "
        f"{'理论均值':>10}  "
        f"{'均值偏差%':>12}  "
        f"{'方差':>10}  "
        f"{'理论方差':>10}  "
        f"{'方差偏差%':>12}  "
        f"{'P5':>8}  "
        f"{'P50':>8}  "
        f"{'P95':>8}"
    )
    lines.append(header)
    lines.append("-" * (name_len + dist_len + 130))

    for e in exps:
        mean_pct_str = f"{e['mean_deviation_pct']:+.2f}%" if e["mean_deviation_pct"] is not None else "  N/A"
        var_pct_str = f"{e['variance_deviation_pct']:+.2f}%" if e["variance_deviation_pct"] is not None else "  N/A"

        line = (
            f"{e['exp_name']:<{name_len}}  "
            f"{e['distribution']:<{dist_len}}  "
            f"{e['num_samples']:>8}  "
            f"{e['mean']:>10.4f}  "
            f"{e['theoretical_mean']:>10.4f}  "
            f"{mean_pct_str:>12}  "
            f"{e['variance']:>10.4f}  "
            f"{e['theoretical_variance']:>10.4f}  "
            f"{var_pct_str:>12}  "
            f"{e['q05']:>8.3f}  "
            f"{e['median']:>8.3f}  "
            f"{e['q95']:>8.3f}"
        )
        lines.append(line)

    lines.append("=" * (name_len + dist_len + 130))
    lines.append(f"共 {len(exps)} 组实验, 输出目录: {summary['output_dir']}")
    return "\n".join(lines)


def export_batch_summary_csv(summary: Dict[str, Any], filepath: str) -> str:
    """
    将批量实验汇总导出为 CSV 文件, 方便后续对比分析或画图。

    参数:
        summary:  run_batch_experiments() 返回的汇总字典
        filepath: 输出 CSV 路径

    返回:
        实际写入的绝对路径
    """
    import os
    abs_path = os.path.abspath(filepath)

    exps = summary["experiments"]
    if not exps:
        raise ValueError("汇总结果为空, 无可导出的数据")

    fieldnames = [
        "exp_name", "distribution", "num_samples", "seed",
        "mean", "theoretical_mean", "mean_deviation_abs", "mean_deviation_pct", "mean_deviation_note",
        "variance", "theoretical_variance", "variance_deviation_abs", "variance_deviation_pct", "variance_deviation_note",
        "std", "min", "max", "q05", "q25", "median", "q75", "q95",
        "sample_csv_path", "report_json_path", "histogram_csv_path",
    ]

    params_keys = set()
    for e in exps:
        params_keys.update(e["params"].keys())
    params_keys = sorted(params_keys)

    with open(abs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["params_" + k for k in params_keys] + fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for e in exps:
            row = {**e}
            for k in params_keys:
                row["params_" + k] = e["params"].get(k, "")
            writer.writerow(row)

    return abs_path


def export_batch_summary_json(summary: Dict[str, Any], filepath: str) -> str:
    """
    将批量实验汇总导出为 JSON 文件。

    参数:
        summary:  run_batch_experiments() 返回的汇总字典
        filepath: 输出 JSON 路径

    返回:
        实际写入的绝对路径
    """
    import os
    abs_path = os.path.abspath(filepath)
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return abs_path


# ============================================================================
# 十二、误差排序视图 + ZIP 打包 + 跨实验对比 CSV
# ============================================================================

def _abs_dev_score(e: Dict[str, Any]) -> float:
    """计算误差综合得分 (均值绝对偏差% + 方差绝对偏差%, 忽略 N/A)。"""
    score = 0.0
    if e.get("mean_deviation_pct") is not None:
        score += abs(e["mean_deviation_pct"])
    if e.get("variance_deviation_pct") is not None:
        score += abs(e["variance_deviation_pct"])
    return score


def format_batch_ranked_table(summary: Dict[str, Any]) -> str:
    """
    生成按综合理论误差从低到高排序的终端对比视图。
    方便一眼看出哪组参数最稳、哪组偏得最厉害。

    误差综合分 = |均值偏差%| + |方差偏差%|（N/A 的项不计入）

    参数:
        summary: run_batch_experiments() 返回的汇总字典

    返回:
        格式化的多行字符串表格 (含排名、误差条可视化)
    """
    exps_raw = summary["experiments"]
    if not exps_raw:
        return "无实验数据"

    exps = sorted(exps_raw, key=_abs_dev_score)

    name_len = max(max(len(e["exp_name"]) for e in exps), 10)
    dist_len = max(max(len(e["distribution"]) for e in exps), 8)
    max_score = max(_abs_dev_score(e) for e in exps) or 1.0

    lines = []
    lines.append("=" * (name_len + dist_len + 110))
    header = (
        f"{'#':>3}  "
        f"{'实验名称':<{name_len}}  "
        f"{'分布':<{dist_len}}  "
        f"{'均值偏差%':>12}  "
        f"{'方差偏差%':>12}  "
        f"{'综合误差':>10}  "
        f"  误差可视化 (0 ~ {max_score:.2f}%)"
    )
    lines.append(header)
    lines.append("-" * (name_len + dist_len + 110))

    for rank, e in enumerate(exps, start=1):
        mean_pct_str = f"{e['mean_deviation_pct']:+.2f}%" if e["mean_deviation_pct"] is not None else "  N/A"
        var_pct_str = f"{e['variance_deviation_pct']:+.2f}%" if e["variance_deviation_pct"] is not None else "  N/A"
        score = _abs_dev_score(e)
        bar_len = int(score / max_score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)

        line = (
            f"{rank:>3}  "
            f"{e['exp_name']:<{name_len}}  "
            f"{e['distribution']:<{dist_len}}  "
            f"{mean_pct_str:>12}  "
            f"{var_pct_str:>12}  "
            f"{score:>9.3f}%  "
            f"  {bar}"
        )
        lines.append(line)

    lines.append("=" * (name_len + dist_len + 110))
    best = exps[0]
    worst = exps[-1]
    lines.append(f"最稳: {best['exp_name']} ({_abs_dev_score(best):.3f}%)   "
                 f"最偏: {worst['exp_name']} ({_abs_dev_score(worst):.3f}%)")
    return "\n".join(lines)


def package_batch_results_zip(
    batch_result: Dict[str, Any],
    zip_path: Optional[str] = None,
    summary_csv_path: Optional[str] = None,
    summary_json_path: Optional[str] = None,
) -> str:
    """
    一键打包: 把所有实验结果 (样本 CSV、报告 JSON、直方图 CSV、汇总表)
    整理到一个 ZIP 文件里, 方便发给别人或归档。

    参数:
        batch_result:       run_batch_experiments() 返回的结果字典
        zip_path:           ZIP 输出路径, None 则自动放在 output_dir 同级
        summary_csv_path:   额外的汇总 CSV 路径 (会一并打包), 可选
        summary_json_path:  额外的汇总 JSON 路径 (会一并打包), 可选

    返回:
        ZIP 文件的绝对路径
    """
    import os
    import zipfile
    from datetime import datetime

    output_dir = batch_result["output_dir"]
    summary = batch_result["summary"]

    if zip_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_dir = os.path.dirname(output_dir) or "."
        zip_path = os.path.join(parent_dir, f"batch_results_{ts}.zip")

    zip_abs = os.path.abspath(zip_path)
    base_dir = os.path.basename(os.path.normpath(output_dir))

    with zipfile.ZipFile(zip_abs, "w", zipfile.ZIP_DEFLATED) as zf:
        # 打包 output_dir 下所有文件
        for root, dirs, files in os.walk(output_dir):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.join(base_dir, os.path.relpath(full, output_dir))
                zf.write(full, arcname)

        # 打包额外的汇总表
        if summary_csv_path and os.path.exists(summary_csv_path):
            zf.write(summary_csv_path, os.path.join(base_dir, os.path.basename(summary_csv_path)))
        if summary_json_path and os.path.exists(summary_json_path):
            zf.write(summary_json_path, os.path.join(base_dir, os.path.basename(summary_json_path)))

        # 生成并写入一份 README.txt 说明
        readme_lines = [
            "随机分布采样引擎 - 批量实验结果包",
            "=" * 50,
            f"生成时间: {summary['timestamp']}",
            f"实验数量: {summary['num_experiments']}",
            f"输出目录: {summary['output_dir']}",
            "",
            "文件说明:",
            "  *_samples.csv     - 原始样本数据",
            "  *_report.json     - 单组实验统计报告",
            "  *_histogram.csv   - 直方图分箱统计",
            "  summary.csv       - 跨实验汇总表 (如果生成)",
            "  summary.json      - 跨实验汇总 (如果生成)",
            "",
            "包含的实验:",
        ]
        for i, e in enumerate(summary["experiments"]):
            readme_lines.append(
                f"  [{i+1}] {e['exp_name']} - {e['distribution']} "
                f"{e['params']} (N={e['num_samples']})"
            )
        zf.writestr(os.path.join(base_dir, "README.txt"), "\n".join(readme_lines))

    return zip_abs


def export_batch_comparison_csv(summary: Dict[str, Any], filepath: str) -> str:
    """
    生成跨实验对比总 CSV, 每行一组实验, 列为:
    实验名、分布、参数(展开成多列)、样本量、种子、
    均值、方差、标准差、min/max、
    分位数 P5/P25/P50/P75/P95、
    理论均值/方差、绝对偏差、百分比偏差、偏差说明

    方便直接导进 Excel / Google Sheets / pandas 继续分析和画图。

    参数:
        summary:  run_batch_experiments() 返回的汇总字典
        filepath: 输出 CSV 路径

    返回:
        实际写入的绝对路径
    """
    import os
    abs_path = os.path.abspath(filepath)

    exps = summary["experiments"]
    if not exps:
        raise ValueError("汇总结果为空, 无可导出的数据")

    # 收集所有出现过的参数键, 展开为独立列
    param_keys = set()
    for e in exps:
        param_keys.update(e["params"].keys())
    param_keys = sorted(param_keys)

    fieldnames = [
        "exp_name", "distribution",
        *[f"param_{k}" for k in param_keys],
        "num_samples", "seed",
        "mean", "theoretical_mean",
        "mean_deviation_abs", "mean_deviation_pct", "mean_deviation_note",
        "variance", "theoretical_variance",
        "variance_deviation_abs", "variance_deviation_pct", "variance_deviation_note",
        "std", "min", "max",
        "q05", "q25", "median", "q75", "q95",
        "sample_csv_path", "report_json_path", "histogram_csv_path",
    ]

    with open(abs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in exps:
            row = {**e}
            for k in param_keys:
                row[f"param_{k}"] = e["params"].get(k, "")
            writer.writerow(row)

    return abs_path
