#!/usr/bin/env python
# ============================================================
# tune_elot_closed.py - ELOT 闭集蒸馏调参脚本 (PCA 投影版)
#
# 放置位置: KD/tune_elot_closed.py
#
# 改进点 (相对于上一版):
#   1. 创建 ELOTClosedLoss 时传入 w_t_full (教师完整分类头权重)
#      用于在初始化时预计算 PCA 投影矩阵
#   2. 其他参数和搜索空间保持不变
#
# 使用方法:
#   # 诊断模式: 跑 3 个 epoch, 观察 C_weight 和 S_teacher 的数量级和 W_ortho_err
#   python tune_elot_closed.py \
#       --path_t ./save/models/ResNet50_.../ResNet50_best.pth \
#       --mode diagnose --epochs 3
#
#   # 单子集调参模式: Optuna 自动搜索
#   python tune_elot_closed.py \
#       --path_t ./save/models/ResNet50_.../ResNet50_best.pth \
#       --mode tune --subset_id 0 --epochs 30 --n_trials 20 \
#       --study_name elot_sub0_round1
#
#   # 全局调参模式 (5个子集平均):
#   python tune_elot_closed.py \
#       --path_t ./save/models/ResNet50_.../ResNet50_best.pth \
#       --mode tune --epochs 30 --n_trials 20 \
#       --study_name elot_global_round1
#
#   # 最终训练: 用找到的最优参数跑完整 240 epoch
#   python tune_elot_closed.py \
#       --path_t ./save/models/ResNet50_.../ResNet50_best.pth \
#       --mode train --epochs 240 \
#       --beta 5.0 --lambda_sem 1.0 --beta_feat 100.0
# ============================================================

from __future__ import print_function

import os
import argparse
import socket
import time
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import tensorboard_logger as tb_logger

from models import model_dict
from models.util import ConvReg
from dataset.cifar100_subset import get_cifar100_closed_subset_dataloaders
from dataset.cifar100_subset import CLASSES_PER_SUBSET
from helper.util import adjust_learning_rate
from helper.loops import validate

from distiller_zoo.ELOT_closed import ELOTClosedLoss, get_classifier_weight
from distiller_zoo.FitNet import HintLoss
from helper.train_elot_closed import train_distill_elot_closed


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_teacher_name(model_path):
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    model_name = get_teacher_name(model_path)
    model = model_dict[model_name](num_classes=n_cls)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    return model


def train_one_subset(subset_id, opt):
    """
    训练一个子集, 返回该子集的 best test accuracy
    """
    print("\n" + "=" * 60)
    print("  子集 {} (类 {}~{})".format(
        subset_id, subset_id * 20, subset_id * 20 + 19))
    print("=" * 60)

    set_random_seed(opt.seed, deterministic=True)

    # ==================== 数据 ====================
    train_loader, val_loader, n_data, class_indices = \
        get_cifar100_closed_subset_dataloaders(
            subset_id=subset_id,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers
        )
    num_classes_student = CLASSES_PER_SUBSET  # 20
    num_classes_teacher = 100

    # ==================== 模型 ====================
    model_t = load_teacher(opt.path_t, num_classes_teacher)
    model_s = model_dict[opt.model_s](num_classes=num_classes_student)

    # ==================== 探测维度 ====================
    data_dummy = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data_dummy, is_feat=True)
    feat_s, _ = model_s(data_dummy, is_feat=True)
    t_dim = feat_t[-1].shape[1]
    s_dim = feat_s[-1].shape[1]

    # ==================== ConvReg (可选) ====================
    use_hint = opt.beta_feat > 0
    conv_reg = None
    if use_hint:
        s_shape = feat_s[opt.hint_layer].shape
        t_shape = feat_t[opt.hint_layer].shape
        conv_reg = ConvReg(s_shape, t_shape)

    # ==================== 提取教师分类头权重 (用于 PCA) ====================
    with torch.no_grad():
        w_t_full_init = get_classifier_weight(model_t).detach().cpu().clone()

    # ==================== ELOT ====================
    elot_criterion = ELOTClosedLoss(
        t_dim=t_dim,
        s_dim=s_dim,
        num_classes_student=num_classes_student,
        class_indices=class_indices,
        w_t_full=w_t_full_init,          # 新增: 用于初始化时计算 PCA
        lambda_sem=opt.lambda_sem,
        epsilon=opt.ot_epsilon,
        warmup_iters=opt.warmup_iters,
        nb_dummies=opt.nb_dummies,
    )

    # ==================== 组装 ====================
    module_list = nn.ModuleList([model_s, elot_criterion])
    trainable_list = nn.ModuleList([model_s, elot_criterion])
    if use_hint:
        module_list.append(conv_reg)
        trainable_list.append(conv_reg)
    module_list.append(model_t)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_list = nn.ModuleList([criterion_cls])
    if use_hint:
        criterion_list.append(HintLoss())

    optimizer = optim.SGD(
        trainable_list.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()

    # ==================== ConvReg 预训练 (简化版) ====================
    if use_hint and opt.init_epochs > 0:
        model_t.eval()
        model_s.eval()
        conv_reg.train()
        reg_optimizer = optim.SGD(
            conv_reg.parameters(), lr=opt.learning_rate,
            momentum=opt.momentum, weight_decay=opt.weight_decay
        )
        hint_criterion = HintLoss().cuda() if torch.cuda.is_available() else HintLoss()

        for ep in range(1, opt.init_epochs + 1):
            for idx, data in enumerate(train_loader):
                input, target, index = data
                input = input.float()
                if torch.cuda.is_available():
                    input = input.cuda()
                with torch.no_grad():
                    feat_s_pre, _ = model_s(input, is_feat=True, preact=False)
                    feat_t_pre, _ = model_t(input, is_feat=True, preact=False)
                    feat_s_pre = [f.detach() for f in feat_s_pre]
                    feat_t_pre = [f.detach() for f in feat_t_pre]
                f_s = conv_reg(feat_s_pre[opt.hint_layer])
                f_t = feat_t_pre[opt.hint_layer]
                loss = hint_criterion(f_s, f_t)
                reg_optimizer.zero_grad()
                loss.backward()
                reg_optimizer.step()
            if ep % 10 == 0 or ep == opt.init_epochs:
                print("  ConvReg 预训练 epoch {}/{}, loss={:.4f}".format(
                    ep, opt.init_epochs, loss.item()))

    # ==================== 正式训练 ====================
    global_iter_counter = [0]
    best_acc = 0

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)

        train_acc, train_loss = train_distill_elot_closed(
            epoch, train_loader, module_list, criterion_list,
            optimizer, opt, global_iter_counter
        )

        test_acc, test_acc_top5, test_loss = validate(
            val_loader, model_s, criterion_cls, opt
        )

        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 10 == 0 or epoch == opt.epochs:
            print("  子集{} Epoch {}/{}: test_acc={:.2f}%, best={:.2f}%".format(
                subset_id, epoch, opt.epochs, test_acc, best_acc))

    print("\n  >>> 子集 {} 最终 best_acc: {:.2f}%".format(subset_id, best_acc))
    return best_acc


def run_all_subsets(opt):
    """依次跑 5 个子集, 汇总结果"""
    print("\n" + "#" * 60)
    print("  ELOT 闭集蒸馏 (PCA 版) - 全部 5 个子集")
    print("  参数: beta={}, beta_feat={}, lambda_sem={}, "
          "warmup_iters={}, ot_epsilon={}".format(
        opt.beta, opt.beta_feat, opt.lambda_sem,
        opt.warmup_iters, opt.ot_epsilon))
    print("#" * 60)

    results = {}
    for subset_id in range(5):
        acc = train_one_subset(subset_id, opt)
        results[subset_id] = acc

    print("\n" + "=" * 60)
    print("  最终结果汇总")
    print("=" * 60)
    for sid in range(5):
        print("  子集 {} (类 {:2d}~{:2d}): {:.2f}%".format(
            sid, sid * 20, sid * 20 + 19, results[sid]))
    avg_acc = np.mean([v.item() if torch.is_tensor(v) else v for v in results.values()])
    print("  " + "-" * 40)
    print("  5 子集平均: {:.2f}%".format(avg_acc))
    print("=" * 60)

    return avg_acc, results


def parse_args():
    parser = argparse.ArgumentParser('ELOT 闭集蒸馏调参 (PCA 版)')

    # 模式
    parser.add_argument('--mode', type=str, default='diagnose',
                        choices=['diagnose', 'tune', 'train'])

    # 基础
    parser.add_argument('--path_t', type=str, required=True)
    parser.add_argument('--model_s', type=str, default='resnet14')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--init_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)

    # 单子集调参
    parser.add_argument('--subset_id', type=int, default=None, choices=[0, 1, 2, 3, 4],
                        help='指定调参的子集编号 (0-4)。若不指定，则调所有子集的平均效果')

    # 优化器
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # 损失权重
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--beta_feat', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=5.0)

    # FitNet
    parser.add_argument('--hint_layer', type=int, default=4)

    # ELOT
    parser.add_argument('--lambda_sem', type=float, default=1.0,
                        help='教师语义相似度权重')
    parser.add_argument('--ot_epsilon', type=float, default=0.1)
    parser.add_argument('--warmup_iters', type=int, default=500)
    parser.add_argument('--nb_dummies', type=int, default=1)

    # Optuna
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--study_name', type=str, default='elot_closed_pca',
                        help='Optuna study 名称，不同调参轮次请用不同名称')
    parser.add_argument('--db_dir', type=str, default='./optuna_db',
                        help='存放 Optuna 数据库的文件夹')

    opt = parser.parse_args()
    opt.lr_decay_epochs = [int(x) for x in opt.lr_decay_epochs.split(',')]

    # 创建数据库文件夹
    if not os.path.isdir(opt.db_dir):
        os.makedirs(opt.db_dir)

    return opt


def main():
    opt = parse_args()

    if opt.mode == 'diagnose':
        print("\n>>> 诊断模式: 跑全部 5 个子集, 每个 {} epochs".format(opt.epochs))
        print(">>> 观察 C_weight, S_teacher, C, W_ortho_err 的数量级和变化\n")
        for sid in range(5):
            train_one_subset(sid, opt)

    elif opt.mode == 'tune':
        try:
            import optuna
        except ImportError:
            print("请先安装 Optuna: python -m pip install optuna")
            sys.exit(1)

        # 确定要调参的子集列表
        if opt.subset_id is not None:
            subsets_to_tune = [opt.subset_id]
            print("\n>>> 单子集调参模式: 子集 {}".format(opt.subset_id))
        else:
            subsets_to_tune = list(range(5))
            print("\n>>> 全局调参模式: 5个子集平均")

        def objective(trial):
            # 搜索超参数
            opt.beta = trial.suggest_float('beta', 0.1, 50.0, log=True)
            opt.lambda_sem = trial.suggest_float('lambda_sem', 0.1, 10.0, log=True)
            opt.ot_epsilon = trial.suggest_float('ot_epsilon', 0.1, 10, log=True)

            print("\n>>> Trial {}: beta={:.3f}, lambda_sem={:.3f}, "
                  "warmup={}, epsilon={:.3f}, beta_feat={:.1f}".format(
                trial.number, opt.beta, opt.lambda_sem,
                opt.warmup_iters, opt.ot_epsilon, opt.beta_feat))

            accs = []
            for subset_id in subsets_to_tune:
                acc = train_one_subset(subset_id, opt)
                acc_val = acc.item() if torch.is_tensor(acc) else acc
                accs.append(acc_val)

                # 多子集模式才报告中间结果和剪枝
                if len(subsets_to_tune) > 1:
                    current_avg = np.mean(accs)
                    trial.report(current_avg, subset_id)

                    if trial.should_prune():
                        print("  >>> Trial {} 被剪枝 (子集{}, 平均={:.2f}%)".format(
                            trial.number, subset_id, current_avg))
                        raise optuna.TrialPruned()

            avg_acc = np.mean(accs)
            for i, sid in enumerate(subsets_to_tune):
                trial.set_user_attr('subset_{}_acc'.format(sid), accs[i])

            return avg_acc

        # 数据库路径: {db_dir}/{study_name}.db
        db_path = 'sqlite:///{}'.format(os.path.join(opt.db_dir, opt.study_name + '.db'))
        print(">>> 数据库路径: {}".format(db_path))

        study = optuna.create_study(
            study_name=opt.study_name,
            direction='maximize',
            storage=db_path,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1,
            ),
        )

        study.optimize(objective, n_trials=opt.n_trials)

        print("\n" + "=" * 60)
        print("  Optuna 调参完成!")
        print("=" * 60)
        print("  最佳 trial: #{}".format(study.best_trial.number))
        print("  最佳准确率: {:.2f}%".format(study.best_value))
        print("  最佳参数:")
        for key, value in study.best_params.items():
            print("    {} = {}".format(key, value))
        print("=" * 60)

        print("\n  最佳 trial 各子集结果:")
        for sid in subsets_to_tune:
            key = 'subset_{}_acc'.format(sid)
            if key in study.best_trial.user_attrs:
                print("    子集 {}: {:.2f}%".format(
                    sid, study.best_trial.user_attrs[sid]))

    elif opt.mode == 'train':
        print("\n>>> 最终训练模式: {} epochs".format(opt.epochs))
        avg_acc, results = run_all_subsets(opt)


if __name__ == '__main__':
    main()
