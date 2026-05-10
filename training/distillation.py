"""Teacher→Student 知识蒸馏损失。

用于 God's-eye training：Teacher 看到完整信息（public + private），
Student 仅看到 public 信息。通过 KL 散度将 Teacher 的策略知识迁移到 Student。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def masked_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    action_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """计算 Teacher→Student 的 KL 散度，仅在合法动作上计算。

    KL(teacher || student) = sum_i teacher_prob_i * log(teacher_prob_i / student_prob_i)

    对非法动作（action_mask=0）施加 -inf mask，使其不参与分布和梯度。

    Args:
        teacher_logits: (B, A) Teacher 输出的 logits
        student_logits: (B, A) Student 输出的 logits
        action_mask: (B, A) 合法动作掩码，1=合法, 0=非法
        temperature: 蒸馏温度，>1 软化分布

    Returns:
        scalar KL 散度（按 batch 平均）
    """
    # 校验 action mask 至少有一个合法动作
    if (action_mask.sum(dim=-1) <= 0).any():
        raise ValueError(
            "action_mask contains samples with no legal actions (all zeros). "
            "Check that oracle trajectory data is not corrupted.")

    # 屏蔽非法动作
    min_val = torch.finfo(teacher_logits.dtype).min
    masked_teacher = teacher_logits.masked_fill(action_mask == 0, min_val)
    masked_student = student_logits.masked_fill(action_mask == 0, min_val)

    # 带温度的 softmax
    teacher_probs = F.softmax(masked_teacher / temperature, dim=-1)
    student_log_probs = F.log_softmax(masked_student / temperature, dim=-1)

    # KL(teacher || student) = sum(teacher_prob * (log_teacher - log_student))
    # 但用 softmax + log_softmax 直接算更稳定
    kl_per_sample = (teacher_probs * (torch.log(teacher_probs + 1e-10)
                                      - student_log_probs)).sum(dim=-1)

    return kl_per_sample.mean()


def masked_kl_loss_with_entropy(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    action_mask: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KL 散度 + Teacher 熵（用于监控）。

    Returns:
        (kl_loss, teacher_entropy)
    """
    if (action_mask.sum(dim=-1) <= 0).any():
        raise ValueError(
            "action_mask contains samples with no legal actions (all zeros). "
            "Check that oracle trajectory data is not corrupted.")

    min_val = torch.finfo(teacher_logits.dtype).min
    masked_teacher = teacher_logits.masked_fill(action_mask == 0, min_val)
    masked_student = student_logits.masked_fill(action_mask == 0, min_val)

    teacher_probs = F.softmax(masked_teacher / temperature, dim=-1)
    student_log_probs = F.log_softmax(masked_student / temperature, dim=-1)

    kl_per_sample = (teacher_probs * (torch.log(teacher_probs + 1e-10)
                                      - student_log_probs)).sum(dim=-1)

    teacher_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-10)).sum(dim=-1)

    return kl_per_sample.mean(), teacher_entropy.mean()


def distillation_loss(
    outputs_teacher: dict,
    outputs_student: dict,
    action_mask: torch.Tensor,
    temperature: float = 2.0,
    alpha_kl: float = 1.0,
    alpha_value: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """完整的蒸馏损失。

    组合：
      - Policy KL: masked_kl_loss(teacher_policy, student_policy)
      - Value MSE: MSE(teacher_value, student_value)
      - Oracle Value: oracle_value 用于 value head 监督

    Args:
        outputs_teacher: Teacher forward 输出字典
        outputs_student: Student forward 输出字典
        action_mask: (B, 77) 合法动作掩码
        temperature: 蒸馏温度
        alpha_kl: KL 损失权重
        alpha_value: 价值蒸馏损失权重

    Returns:
        (total_loss, loss_components_dict)
    """
    loss_kl = masked_kl_loss(
        outputs_teacher['policy_logits'],
        outputs_student['policy_logits'],
        action_mask,
        temperature=temperature,
    )

    loss_value = F.mse_loss(
        outputs_student['value'],
        outputs_teacher.get('oracle_value', outputs_teacher['value']),
    )

    total = alpha_kl * loss_kl + alpha_value * loss_value

    components = {
        'distill/kl': loss_kl.item(),
        'distill/value_mse': loss_value.item(),
        'distill/total': total.item(),
    }

    if 'oracle_value' in outputs_teacher:
        components['distill/teacher_oracle_value_mean'] = \
            outputs_teacher['oracle_value'].mean().item()

    return total, components
# 中文注释：Teacher→Student 知识蒸馏工具，包含 Masked KL 散度和价值蒸馏损失。
