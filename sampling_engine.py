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
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar
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
    else:
        return _sample_poisson_large(lam, rng)


def _sample_poisson_small(lam: float, rng: UniformRNG) -> int:
    """Knuth 算法: 适用于 λ 较小 (< 30) 的情形。"""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random_open()
        if p < L:
            return k - 1


def _sample_poisson_large(lam: float, rng: UniformRNG) -> int:
    """
    正态近似 + 拒绝采样: 适用于 λ 较大 (>= 30) 的情形。
    (基于 Ahrens & Dieter 的方法)
    """
    c = 0.767 - 3.36 / lam
    beta = math.pi / math.sqrt(3.0 * lam)
    alpha = beta * lam
    k = math.log(c) - lam - math.log(beta)

    while True:
        u1 = rng.random_open()
        x = (alpha - math.log((1.0 - u1) / u1)) / beta
        n = int(math.floor(x + 0.5))
        if n < 0:
            continue
        u2 = rng.random_open()
        if alpha + beta * x - math.log(u2) <= k + n * math.log(lam) - math.lgamma(n + 1):
            return n


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
            probabilities: 离散概率分布, 长度为 n, 每个元素 >= 0, 和为 1
            rng: 均匀随机数生成器 (可选)
        """
        probs = list(probabilities)
        n = len(probs)
        if n == 0:
            raise ValueError("概率分布不能为空")

        total = sum(probs)
        if total == 0:
            raise ValueError("概率之和不能为零")
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
    """
    n = len(items)
    if k > n:
        raise ValueError("无放回抽样时 k 不能超过物品总数")

    keys = []
    for i in range(n):
        if weights[i] == 0:
            keys.append((0.0, i))
        else:
            u = rng.random_open()
            key = math.log(u) / weights[i]
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


def sample_beta(alpha: float, beta: float, rng: Optional[UniformRNG] = None) -> float:
    """Beta 分布采样 (基于两个 Gamma 分布)。"""
    rng = rng or _default_rng
    x = _sample_gamma(alpha, 1.0, rng)
    y = _sample_gamma(beta, 1.0, rng)
    return x / (x + y)


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
    n: int,
    *args,
    **kwargs,
) -> List[T]:
    """
    批量采样工具: 调用采样函数 n 次并返回结果列表。

    参数:
        sampler_fn: 采样函数 (如 sample_normal_boxmuller)
        n:          样本数量
        *args:      传递给采样函数的位置参数
        **kwargs:   传递给采样函数的关键字参数

    返回:
        长度为 n 的样本列表

    示例:
        samples = batch_sample(sample_normal_boxmuller, 10000, mu=0, sigma=1)
    """
    return [sampler_fn(*args, **kwargs) for _ in range(n)]
