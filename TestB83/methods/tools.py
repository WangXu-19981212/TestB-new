import numpy as np
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

import torch

START_EPS = 16 / 255


def calculate_energy_ratio(features):
    # 将张量从 CUDA 设备移动到 CPU（如果需要）
    if features.is_cuda:
        features = features.cpu()

    # 进行傅里叶变换
    f_transform = torch.fft.fft2(features)
    f_transform_shifted = torch.fft.fftshift(f_transform)

    # 计算幅度谱
    magnitude_spectrum = torch.abs(f_transform_shifted)

    # 计算总能量
    total_energy = torch.sum(magnitude_spectrum ** 2)

    # 设计低通滤波器和高通滤波器
    rows, cols = features.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    mask_low = torch.zeros((rows, cols), dtype=torch.uint8)
    mask_high = torch.ones((rows, cols), dtype=torch.uint8)

    # 低通滤波器
    mask_low[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # 高通滤波器
    mask_high[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # 应用滤波器
    low_freq_energy = torch.sum((magnitude_spectrum * mask_low) ** 2)
    high_freq_energy = torch.sum((magnitude_spectrum * mask_high) ** 2)

    # 计算能量占比
    low_freq_ratio = low_freq_energy / total_energy
    high_freq_ratio = high_freq_energy / total_energy

    return low_freq_ratio.item(), high_freq_ratio.item()


def loss_fn_example(phase):
    # 假设一个简单的正则化目标
    return torch.sum(phase ** 2)


def reparameterize( mu, std, epsilon_norm):
    """实现重参数化技巧"""
    return mu + std * epsilon_norm


# 1. 高频和低频分量提取
def extract_high_low_frequency_components(feature):
    """
    使用 DWT 前向变换提取高频和低频分量。

    参数：
    feature (Tensor): 输入特征图，形状为 (N, C, H, W)，N 为批量大小，C 为通道数，H 和 W 为特征图的高度和宽度。

    返回：
    low_freq (Tensor): 低频分量。
    high_freq (Tensor): 高频分量。
    """
    # 使用 DWT 前向变换，选择小波类型（例如 'haar'）和小波的级数（例如 1）
    dwt = DWTForward(J=1, wave='haar', mode='zero').cuda()

    # 输入特征图进行小波变换，返回四个小波系数：LL、LH、HL、HH
    LL, HH = dwt(feature)

    # 返回低频和高频分量
    low_freq = LL  # 低频分量
    high_freq = HH[0]
    # print("high_freq 的类型:", type(high_freq))

    return low_freq, high_freq


def add_noise_to_freq(freq, noise_type="gaussian", noise_scale=1.0, eps=1e-8):
    """
    在低频分量 (LL) 上添加噪声，支持全局或局部加噪。

    参数：
    freq (Tensor): 低频分量张量，形状为 (B, C, H, W)
    noise_type (str): 噪声类型，可选 "gaussian" 或 "uniform"
    noise_scale (float): 噪声强度缩放因子
    eps (float): 防止除零的小量

    返回：
    Tensor: 加噪后的低频分量
    """
    B, C, H, W = freq.shape

    # 计算全局均值和标准差
    mu = torch.mean(freq, dim=(2, 3), keepdim=True)
    var = torch.var(freq, dim=(2, 3), keepdim=True)
    std = (var + eps).sqrt()

    # 生成噪声
    if noise_type == "gaussian":
        epsilon_mu = torch.randn_like(mu) * noise_scale  # 高斯噪声
        epsilon_std = torch.randn_like(std) * noise_scale
    elif noise_type == "uniform":
        epsilon_mu = (torch.rand_like(mu) * 2 - 1.) * noise_scale  # 均匀噪声 [-scale, scale]
        epsilon_std = (torch.rand_like(std) * 2 - 1.) * noise_scale
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    # 重参数化：调整均值和标准差
    beta = mu + epsilon_mu * std  # 扰动均值
    gamma = std + epsilon_std * std  # 扰动标准差

    # 应用仿射变换
    noisy_freq = gamma * (freq - mu) / std + beta

    return noisy_freq

def pgd_attack(init_input, epsilon, data_grad, num_steps, START_EPS=0.001):
    """
    Projected Gradient Descent (PGD) 对抗样本生成函数（隐式方法）。

    Args:
        init_input: 原始输入数据（Tensor）。
        epsilon: 扰动的最大范围（L∞ 范数约束）。
        data_grad: 输入数据的梯度（Tensor）。
        num_steps: 迭代次数。
        step_size: 每次迭代的步长。
        START_EPS: 随机初始化的范围（默认 0.001）。

    Returns:
        adv_input: 生成的对抗样本。
    """
    adv_input = init_input
    # 随机初始化
    adv_input = init_input + torch.randn_like(init_input) * START_EPS
    step_size = epsilon / num_steps

    # 迭代生成对抗样本
    for _ in range(num_steps):
        # 获取梯度符号
        sign_data_grad = data_grad.sign()

        # 更新对抗样本
        adv_input = adv_input + step_size * sign_data_grad

        # 投影到 epsilon 范围内
        adv_input = torch.max(torch.min(adv_input, init_input + epsilon), init_input - epsilon)

    return adv_input


# 2. 添加扰动
def fgsm_attack(init_input, epsilon, data_grad):
    # print("init_input.shape",init_input.shape)
    init_input = init_input + torch.randn_like(init_input) * 1

    sign_data_grad = data_grad.sign()
    adv_input = init_input + 1 * epsilon * sign_data_grad
    # print("adv_input.shape",adv_input.shape)
    return adv_input


# 3. 逆变换
def reconstruct_feature(low_freq, high_freq):
    # print("high_freq 的类型:", type(high_freq))

    idwt = DWTInverse(wave='haar', mode='zero').cuda()
    # 确保 high_freq 是列表，如果只是一个张量，需包装为列表
    if not isinstance(high_freq, list):
        high_freq = [high_freq]
    # 将高低频分量重建成特征
    reconstructed_feature = idwt((low_freq, high_freq))

    # 返回重构后的特征图的实部
    return reconstructed_feature


def mutual_attention(q, k):
    """
    复杂互注意力机制：结合加权和规范化操作，支持更灵活的关系建模。

    Args:
        q: 查询张量（Tensor）。形状为 [batch_size, seq_len, d_model]
        k: 键张量（Tensor）。形状为 [batch_size, seq_len, d_model]

    Returns:
        v: 加权后的输出张量，形状与 k 相同。
    """
    assert q.size() == k.size(), "q and k must have the same shape"

    # 计算注意力权重（逐元素相乘并进行规范化）
    weight = q * k

    # 规范化权重：沿着特定维度进行 softmax
    weight_norm = F.softmax(weight, dim=-1)

    # 使用规范化权重加权键张量
    v = weight_norm * k

    # 可选：引入额外的非线性激活函数（如 ReLU 或 GELU）
    v = F.relu(v)

    return v


def consistency_loss(scoresM1, scoresM2, type='euclidean'):
    if (type == 'euclidean'):
        avg_pro = (scoresM1 + scoresM2) / 2.0
        matrix1 = torch.sqrt(torch.sum((scoresM1 - avg_pro) ** 2, dim=1))
        matrix2 = torch.sqrt(torch.sum((scoresM2 - avg_pro) ** 2, dim=1))
        dis1 = torch.mean(matrix1)
        dis2 = torch.mean(matrix2)
        dis = (dis1 + dis2) / 2.0
    elif (type == 'KL1'):
        avg_pro = (scoresM1 + scoresM2) / 2.0
        matrix1 = torch.sum(
            F.softmax(scoresM1, dim=-1) * (F.log_softmax(scoresM1, dim=-1) - F.log_softmax(avg_pro, dim=-1)), 1)
        matrix2 = torch.sum(
            F.softmax(scoresM2, dim=-1) * (F.log_softmax(scoresM2, dim=-1) - F.log_softmax(avg_pro, dim=-1)), 1)
        dis1 = torch.mean(matrix1)
        dis2 = torch.mean(matrix2)
        dis = (dis1 + dis2) / 2.0
    elif (type == 'KL2'):
        matrix = torch.sum(
            F.softmax(scoresM2, dim=-1) * (F.log_softmax(scoresM2, dim=-1) - F.log_softmax(scoresM1, dim=-1)), 1)
        dis = torch.mean(matrix)
    elif (type == 'KL3'):
        matrix = torch.sum(
            F.softmax(scoresM1, dim=-1) * (F.log_softmax(scoresM1, dim=-1) - F.log_softmax(scoresM2, dim=-1)), 1)
        dis = torch.mean(matrix)
    else:
        return
    return dis
