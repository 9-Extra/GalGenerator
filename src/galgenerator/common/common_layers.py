import einops
import torch
from torch import nn, Tensor
from torch.nn import Module, functional


# 3x3卷积
class Conv1(Module):
    def __init__(self, in_channel: int, out_channel: int, group: int = 32, active=True):
        super().__init__()
        self.active = torch.nn.SiLU(True) if active else torch.nn.Identity()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=True)
        self.norm = nn.GroupNorm(group, out_channel)
        # self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.active(self.norm(self.conv(x)))
        return x


# 2个 3x3卷积
class Conv2(Module):
    def __init__(self, in_channel: int, out_channel: int, group: int = 32):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(group, out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(group, out_channel)

    def forward(self, x: Tensor) -> Tensor:
        x = functional.silu(self.norm1(self.conv1(x)), inplace=True)
        x = functional.silu(self.norm2(self.conv2(x)), inplace=True)
        return x


class ScaleDotAttention(Module):
    """
    纯纯的Attention层
    """
    scale: float

    def __init__(self, dim: int):
        """
        :param dim:  请求向量的长度，用于计算缩放系数
        """
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        计算结果，没有可以训练的参数
        :param q: query向量[B, 请求个数c，请求向量长度d]
        :param k: key向量[B, key-value的个数n，请求向量长度d]
        :param v: value向量[B, key-value的个数, value向量长度v]
        :return: 查询结果[B, 请求个数, value向量长度v]
        """
        # query与key内积
        q = q * self.scale  # 缩放
        score = torch.einsum("bcd, bnd -> bcn", q, k)  # 每一个请求中每一个向量的相关性
        score = torch.nn.functional.softmax(score, dim=-1)  # 对于相关性进行归一化
        x = torch.einsum("bcn, bnv -> bcv", score, v)

        return x


class MultiHeadAttention(Module):
    """
    多头Attention层
    """
    head: int
    scale: float  # 缩放系数

    q_mlp: Module  # 使用一个线性层计算所有head的投影
    k_mlp: Module
    v_mlp: Module
    out_mlp: Module

    def __init__(self, q_dim: int, v_dim: int, out_dim: int, head: int = 8, head_dim: int = 32):
        """
        :param out_dim: 输出的特征数
        :param head: head数
        :param head_dim: 每个head的隐藏层向量的长度
        """
        super().__init__()
        self.head = head

        self.scale = head_dim ** -0.5
        hidden_dim = head * head_dim  # 所有head的隐藏层加起来的维数
        self.q_mlp = torch.nn.Linear(q_dim, hidden_dim)
        self.k_mlp = torch.nn.Linear(v_dim, hidden_dim)
        self.v_mlp = torch.nn.Linear(v_dim, hidden_dim)
        self.out_mlp = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        :param q: query向量[B, 请求个数c，请求向量长度d]
        :param k: key向量[B, key-value的个数n，请求向量长度d]
        :param v: value向量[B, key-value的个数, value向量长度v]
        :return: 查询结果[B, 请求个数, value向量长度v]
        """
        q, k, v = self.q_mlp(q), self.k_mlp(k), self.v_mlp(v)
        q, k, v = [einops.rearrange(v, "b(hc)d -> bhcd", h=self.head) for v in [q, k, v]]

        # query与key内积
        q = q * self.scale  # 缩放
        score = torch.einsum("bhcd, bhnd -> bhcn", q, k)  # 每一个请求中每一个向量的相关性
        score = torch.nn.functional.softmax(score, dim=-1)  # 对于相关性进行归一化
        x = torch.einsum("bhcn, bhnv -> bhcv", score, v)
        x = einops.rearrange(x, "bhcv -> h(hc)v")
        return self.out_mlp(x)


class MultiHeadSelfAttention(Module):
    """
    多头自注意力层
    """
    head: int
    scale: float  # 缩放系数

    q_mlp: Module  # 使用一个线性层计算所有head的投影
    k_mlp: Module
    v_mlp: Module
    out_mlp: Module

    def __init__(self, in_dim: int, out_dim: int, head: int = 8, head_dim: int = 32):
        """
        :param in_dim: 输入的特征数
        :param out_dim: 输出的特征数
        :param head: head数
        :param head_dim: 每个head的隐藏层向量的长度
        """
        super().__init__()
        self.head = head

        self.scale = head_dim ** -0.5
        hidden_dim = head * head_dim  # 所有head的隐藏层加起来的维数
        self.q_mlp = torch.nn.Linear(in_dim, hidden_dim)
        self.k_mlp = torch.nn.Linear(in_dim, hidden_dim)
        self.v_mlp = torch.nn.Linear(in_dim, hidden_dim)
        self.out_mlp = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor):
        q, k, v = self.q_mlp(x), self.k_mlp(x), self.v_mlp(x)
        q, k, v = [einops.rearrange(v, "b(hc)d -> bhcd", h=self.head) for v in [q, k, v]]

        # query与key内积
        q = q * self.scale  # 缩放
        score = torch.einsum("bhcd, bhnd -> bhcn", q, k)  # 每一个请求中每一个向量的相关性
        score = torch.nn.functional.softmax(score, dim=-1)  # 对于相关性进行归一化
        x = torch.einsum("bhcn, bhnv -> bhcv", score, v)
        x = einops.rearrange(x, "bhcv -> h(hc)v")
        return self.out_mlp(x)


class MultiHeadSelfAttentionCV(Module):
    """
    多头自注意力层，使用1x1卷积而非线性层进行投影，另外假定输入是图像形式的四维张量
    好像就是non local attention
    """
    head: int
    scale: float  # 缩放系数

    qkv: Module  # 使用一个卷积层计算所有head的投影
    out: Module  # 输出时也使用卷积进行投影

    def __init__(self, in_dim: int, out_dim: int, head: int = 8, head_dim: int = 32):
        """
        :param in_dim: 输入的特征数
        :param out_dim: 输出的特征数
        :param head: head数
        :param head_dim: 每个head的隐藏层向量的长度
        """
        super().__init__()
        self.head = head

        self.scale = head_dim ** -0.5
        hidden_dim = head * head_dim  # 所有head的隐藏层加起来的维数
        self.qkv = torch.nn.Conv2d(in_dim, hidden_dim * 3, 1, bias=False)
        self.out = torch.nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        :param x: 四维张量[batch, channel, x, y]
        """
        _, _, w, h = x.shape  # 用于还原

        q, k, v = self.qkv(x).chunk(3, dim=1)
        q, k, v = [einops.rearrange(v, "b (h c) x y -> b h (x y) c", h=self.head) for v in [q, k, v]]

        # query与key内积（在空间位置维度上计算注意力）
        q = q * self.scale  # 缩放
        score = torch.einsum("bhqc, bhkc -> bhqk", q, k)  # 空间位置之间的相关性
        score = torch.nn.functional.softmax(score, dim=-1)  # 对于相关性进行归一化
        x = torch.einsum("bhqk, bhkc -> bhqc", score, v)
        x = einops.rearrange(x, "b h (x y) c -> b (h c) x y", x=w, y=h)
        return self.out(x)


class LinearAttentionCV(Module):
    """
    线性多头注意力层（Linear Attention），使用1x1卷积进行投影。

    原理：
        标准 Attention:  out = softmax(Q @ K^T / sqrt(d)) @ V
                         复杂度 O(seq_len^2)

        Linear Attention: out = (Q' @ K'^T) @ V = Q' @ (K'^T @ V)
                         其中 Q' = softmax(Q, dim=-1), K' = softmax(K, dim=-1)
                         复杂度 O(seq_len)

    计算步骤：
        1. 将 Q, K 在特征维度 (head_dim) 上做 softmax
        2. 先算 K^T @ V: [head_dim, head_dim] 矩阵
        3. 再算 Q @ (K^T @ V): [seq_len, head_dim]
        4. 除以 head_dim 归一化

    相比标准 Attention 的优势：
        - 复杂度从 O(N^2) 降到 O(N)，适合长序列（如 32x32=1024）
        - 效果略低于标准 Attention，但远好于无 Attention
    """
    head: int
    head_dim: int

    qkv: Module  # 使用一个卷积层计算所有head的投影
    out: Module  # 输出时也使用卷积进行投影

    def __init__(self, in_dim: int, out_dim: int, head: int = 8, head_dim: int = 32):
        """
        :param in_dim: 输入的特征数
        :param out_dim: 输出的特征数
        :param head: head数
        :param head_dim: 每个head的隐藏层向量的长度
        """
        super().__init__()
        self.head = head
        self.head_dim = head_dim
        hidden_dim = head * head_dim  # 所有head的隐藏层加起来的维数
        self.qkv = torch.nn.Conv2d(in_dim, hidden_dim * 3, 1, bias=False)
        self.out = torch.nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        :param x: 四维张量[batch, channel, x, y]
        :return: 同形状的四维张量
        """
        _, _, w, h = x.shape  # 用于还原

        q, k, v = self.qkv(x).chunk(3, dim=1)
        q, k, v = [einops.rearrange(t, "b (h c) x y -> b h (x y) c", h=self.head) for t in [q, k, v]]

        # 在特征维度上做 softmax，保证可分解性
        q = q.softmax(dim=-1)  # [batch, head, seq_len, head_dim]
        k = k.softmax(dim=-1)  # [batch, head, seq_len, head_dim]

        # Linear Attention 核心: 先算 K^T @ V，再算 Q @ (K^T @ V)
        # kv: [batch, head, head_dim, head_dim]
        kv = torch.einsum("b h n d, b h n e -> b h d e", k, v)
        # out: [batch, head, seq_len, head_dim]
        out = torch.einsum("b h n d, b h d e -> b h n e", q, kv)

        # 除以 head_dim 归一化，防止数值过大
        out = out / self.head_dim

        out = einops.rearrange(out, "b h (x y) c -> b (h c) x y", x=w, y=h)
        return self.out(out)


class ResBlock(Module):
    """
    残差块，包含两个3x3卷积块，卷积块内使用group normal
    """
    conv1: Module
    conv2: Module
    adjust: Module

    def __init__(self, in_dim: int, out_dim: int, group: int = 32):
        super().__init__()
        self.conv1 = Conv1(in_dim, out_dim, group)
        self.conv2 = Conv1(out_dim, out_dim, group)
        self.adjust = torch.nn.Identity() if in_dim == out_dim else torch.nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x: torch.Tensor):
        return self.adjust(x) + self.conv2(self.conv1(x))


class TimeEmbedResBlock(Module):
    """
    进行时间嵌入残差块，包含两个3x3卷积块，卷积块内使用group normal
    """
    out_dim: int

    conv1: Module
    conv2: Module
    mlp: Module
    adjust: Module

    def __init__(self, in_dim: int, out_dim: int, time_embed_dim: int, group: int = 32):
        super().__init__()
        self.out_dim = out_dim

        self.conv1 = Conv1(in_dim, out_dim, group)
        self.conv2 = Conv1(out_dim, out_dim, group)
        self.mlp = torch.nn.Linear(time_embed_dim, out_dim * 2)
        self.adjust = torch.nn.Identity() if in_dim == out_dim else torch.nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        h = self.conv1(x)
        scale, shift = self.mlp(time_emb).unsqueeze_(-1).unsqueeze_(-1).chunk(2, dim=1)
        h = h * (scale + 1) + shift
        h = self.conv2(h)
        return self.adjust(x) + h


class ChannelDownSample(Module):
    """
    通过将图像切成4块再叠到channel维，最后一个1x1卷积得到输出维数的下采样层
    """
    conv: Module

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_dim * 4, out_dim, 1)

    def forward(self, x: torch.Tensor):
        x = einops.rearrange(x, "b c (x s1) (y s2) -> b (c s1 s2) x y", s1=2, s2=2)
        return self.conv(x)
