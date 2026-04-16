# ============================================================
# ELOT_closed.py - 闭集 ELOT 蒸馏损失 (PCA 投影版)
#
# 放置位置: KD/distiller_zoo/ELOT_closed.py
#
# 核心思路 (相对于上一版的改进):
#   上一版: 学生 (20, 64) → proj (64→2048) → (20, 2048)
#          131,072 参数对齐 20 个向量, 投影层过拟合, ELOT 饱和
#
#   本版:   教师 (20, 2048) → PCA 固定 (2048→64) → (20, 64)  无损降维
#          学生 (20, 64)   → W 可训练 (64→64)    → (20, 64)  方阵旋转
#          4,096 参数, 且 W 是严格正交矩阵, 防止作弊
#
# 关键设计:
#   1. 教师侧用 PCA 降维到 s_dim, 利用"20 个向量最多占 20 维"的性质
#      PCA 投影零信息损失 (因为 s_dim=64 >> 20)
#   2. 学生侧用方阵投影 (s_dim, s_dim), 正交初始化保证纯旋转
#      方阵的维度和学生原始维度一致, 不丢失学生权重的任何方向信息
#   3. 代价矩阵: C = -cos(proj(w_s), w_t_pca) - λ * S_teacher
#      S_teacher 在原始 2048 维空间算, 保留完整的类间语义结构
# ============================================================

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
from ot.lp import emd


# ============================================================
# ELOT 求解器 (不变)
# ============================================================

def elot_emd(a, b, M, nb_dummies=1, log=False, **kwargs):
    b_extended = np.append(b, [(np.sum(a)) / nb_dummies] * nb_dummies)
    a_extended = np.append(a, [(np.sum(b)) / nb_dummies] * nb_dummies)
    M_extended = np.zeros((len(a_extended), len(b_extended)))
    M_extended[:len(a), :len(b)] = M

    gamma, log_ot = emd(a_extended, b_extended, M_extended, log=True, **kwargs)

    if log_ot['warning'] is not None:
        raise ValueError("EMD 求解出错: 尝试增加 nb_dummies 的值")

    log_ot['partial_w_dist'] = np.sum(M * gamma[:len(a), :len(b)])

    if log:
        return gamma[:len(a), :len(b)], log_ot
    else:
        return gamma[:len(a), :len(b)]


def elot_entropic(a, b, M, reg, nb_dummies=1, numItermax=1000,
                  stopThr=1e-100, verbose=False, log=False, **kwargs):
    b_extended = np.append(b, [(np.sum(a)) / nb_dummies] * nb_dummies)
    a_extended = np.append(a, [(np.sum(b)) / nb_dummies] * nb_dummies)
    M_extended = np.zeros((len(a_extended), len(b_extended)))
    M_extended[:len(a), :len(b)] = M

    gamma, log_ot = ot.sinkhorn(
        a_extended, b_extended, M_extended, reg,
        numItermax=numItermax, stopThr=stopThr,
        verbose=verbose, log=True, **kwargs
    )

    log_ot['partial_w_dist'] = np.sum(M * gamma[:len(a), :len(b)])

    if log:
        return gamma[:len(a), :len(b)], log_ot
    else:
        return gamma[:len(a), :len(b)]


# ============================================================
# 工具函数: 从模型中提取分类头权重
# ============================================================

def get_classifier_weight(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        return model.fc.weight
    elif hasattr(model, 'linear') and isinstance(model.linear, nn.Linear):
        return model.linear.weight
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            for m in model.classifier:
                if isinstance(m, nn.Linear):
                    return m.weight
        elif isinstance(model.classifier, nn.Linear):
            return model.classifier.weight

    raise ValueError(
        "无法自动提取分类头权重, 请检查模型结构。"
        "支持的属性名: model.fc, model.linear, model.classifier"
    )


# ============================================================
# ELOTClosedLoss — PCA 投影版
# ============================================================

class ELOTClosedLoss(nn.Module):
    """
    闭集 ELOT 蒸馏损失 (PCA 投影版)

    投影设计:
      教师 (20, t_dim) → PCA 固定投影 (t_dim → s_dim) → (20, s_dim)
        - PCA 是无损降维 (20 个向量最多占 20 维, s_dim >> 20)
        - 完全不需要训练, 用 register_buffer 存储

      学生 (20, s_dim) → 可训练方阵 W (s_dim → s_dim) → (20, s_dim)
        - 方阵, 正交初始化, 纯旋转变换
        - 参数量小 (s_dim²), 不容易过拟合

    注意: 要求学生特征维度 s_dim >= 20 (K)
    对于 resnet14 (s_dim=64), 自然满足
    """

    def __init__(self, t_dim, s_dim, num_classes_student,
                 class_indices, w_t_full,
                 lambda_sem=1.0, epsilon=0.1,
                 warmup_iters=500, nb_dummies=1):
        """
        参数:
            t_dim: int, 教师特征维度
            s_dim: int, 学生特征维度 (要求 >= num_classes_student)
            num_classes_student: int, 学生类别数 (=20)
            class_indices: list, 学生对应教师的原始类别编号
            w_t_full: Tensor (100, t_dim), 教师完整分类头权重
                初始化时用于计算 PCA 投影矩阵
            lambda_sem: float, 教师语义相似度奖励项的权重
            epsilon: float, OT 熵正则化系数
            warmup_iters: int, 标准 OT warmup 迭代数
            nb_dummies: int, ELOT 虚拟点数量
        """
        super(ELOTClosedLoss, self).__init__()

        K = num_classes_student
        assert len(class_indices) == K, \
            "class_indices 长度 ({}) 必须等于 num_classes_student ({})".format(
                len(class_indices), K)
        assert s_dim >= K, \
            "s_dim ({}) 必须 >= num_classes_student ({}) 才能保证 PCA 无损".format(
                s_dim, K)

        # ===== 用教师的 20 类权重做 PCA =====
        with torch.no_grad():
            w_t_full = w_t_full.float()
            class_indices_tensor = torch.LongTensor(class_indices)
            w_t = w_t_full[class_indices_tensor]  # (20, t_dim)

            # 中心化 (PCA 标准步骤)
            pca_mean = w_t.mean(dim=0, keepdim=True)  # (1, t_dim)
            w_t_centered = w_t - pca_mean

            # SVD 分解
            U, S, V = torch.svd(w_t_centered)  # V: (t_dim, min(20, t_dim))

            # V 实际有 min(K, t_dim) 列 (通常是 20)
            # 我们需要 s_dim (通常是 64) 列, 多出来的用 0 填充
            num_real_dims = V.shape[1]

            if num_real_dims >= s_dim:
                pca_proj = V[:, :s_dim]
            else:
                # num_real_dims=20 < s_dim=64 时的情况
                pca_proj = torch.zeros(t_dim, s_dim)
                pca_proj[:, :num_real_dims] = V

            # 预计算教师在 PCA 空间中的坐标 (永不改变)
            w_t_pca = (w_t - pca_mean) @ pca_proj  # (20, s_dim)

            # 预计算教师自身的语义相似度 (在原始 t_dim 空间, 零信息损失)
            w_t_norm = F.normalize(w_t, dim=1)
            S_teacher = w_t_norm @ w_t_norm.t()  # (20, 20)

        # 注册为 buffer: 不训练, 但跟着模块 .cuda()
        self.register_buffer('pca_proj', pca_proj)        # (t_dim, s_dim)
        self.register_buffer('pca_mean', pca_mean.squeeze(0))  # (t_dim,)
        self.register_buffer('w_t_pca', w_t_pca)          # (20, s_dim)
        self.register_buffer('S_teacher', S_teacher)      # (20, 20)
        self.register_buffer('class_indices',
                             torch.LongTensor(class_indices))

        # ===== 学生可训练投影层: 方阵, 正交初始化 =====
        self.proj_s = nn.Linear(s_dim, s_dim, bias=False)
        nn.init.orthogonal_(self.proj_s.weight)

        # 超参数
        self.lambda_sem = lambda_sem
        self.epsilon = epsilon
        self.warmup_iters = warmup_iters
        self.nb_dummies = nb_dummies
        self.num_classes_student = K
        self.s_dim = s_dim
        self.t_dim = t_dim

        # 打印配置信息
        print("\n" + "=" * 60)
        print("ELOT 投影层配置 (PCA 版):")
        print("  教师 PCA: {} → {} (无损降维, 因为 20 个向量最多占 20 维)".format(
            t_dim, s_dim))
        print("  学生投影: {} → {} (方阵, 正交初始化, 严格正交)".format(
            s_dim, s_dim))
        print("  PCA 奇异值前 5 个: {}".format(
            [round(x, 3) for x in S[:min(5, len(S))].tolist()]))
        print("  PCA 奇异值最后 3 个: {}".format(
            [round(x, 4) for x in S[-3:].tolist()]))
        print("  学生投影层参数量: {:,}".format(s_dim * s_dim))
        print("=" * 60 + "\n")

    def forward(self, w_s, w_t_full, current_iter):
        """
        计算闭集 ELOT 蒸馏损失

        参数:
            w_s: (20, s_dim) 学生分类头权重 (保留梯度)
            w_t_full: (100, t_dim) 教师分类头权重
                注意: PCA 投影在初始化时已预计算, 本版不使用这个参数
                保留参数只是为了和训练循环接口兼容
            current_iter: int, 当前全局迭代步数

        返回:
            loss: 标量 tensor
        """
        K = self.num_classes_student
        device = w_s.device

        # ===== 学生投影到共享空间 (方阵旋转) =====
        w_s_proj = self.proj_s(w_s)  # (20, s_dim)

        # ===== C_weight: 余弦相似度的负值 =====
        w_t_pca_norm = F.normalize(self.w_t_pca, dim=1)   # (20, s_dim)
        w_s_proj_norm = F.normalize(w_s_proj, dim=1)       # (20, s_dim)

        # 余弦相似度矩阵 (20, 20)
        cos_ts = w_t_pca_norm @ w_s_proj_norm.t()
        C_weight = -cos_ts

        # ===== 代价矩阵 =====
        C = C_weight - self.lambda_sem * self.S_teacher

        # 诊断打印
        if current_iter % 50 == 0:
            neg_ratio = (C < 0).float().mean().item()
            # 检查投影层是否仍接近正交
            W = self.proj_s.weight
            ortho_error = (W @ W.t() - torch.eye(
                self.s_dim, device=W.device)).abs().mean().item()

            print("  [ELOT] iter={} C_weight: mean={:.4f} min={:.4f} max={:.4f} | "
                  "S_teacher: diag={:.4f} offdiag={:.4f} | "
                  "C: mean={:.4f} min={:.4f} max={:.4f} neg={:.1%} | "
                  "W_ortho_err={:.4f}".format(
                current_iter,
                C_weight.mean().item(),
                C_weight.min().item(),
                C_weight.max().item(),
                self.S_teacher.diag().mean().item(),
                (self.S_teacher.sum() - self.S_teacher.diag().sum()).item() / (K * K - K),
                C.mean().item(), C.min().item(), C.max().item(),
                neg_ratio, ortho_error))

        # ===== OT 求解 =====
        a = ot.unif(K)
        b = ot.unif(K)
        C_cpu = C.detach().cpu().numpy()

        if current_iter <= self.warmup_iters:
            if self.epsilon == 0:
                pi = ot.emd(a, b, C_cpu)
            else:
                pi = ot.sinkhorn(a, b, C_cpu, reg=self.epsilon)
        else:
            if self.epsilon == 0:
                pi = elot_emd(a, b, C_cpu, nb_dummies=self.nb_dummies)
            else:
                pi = elot_entropic(a, b, C_cpu, reg=self.epsilon,
                                   nb_dummies=self.nb_dummies)

        # ===== 传输损失 =====
        pi_tensor = torch.from_numpy(pi).float().to(device)
        loss = torch.sum(pi_tensor * C)

        return loss