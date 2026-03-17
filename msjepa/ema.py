from __future__ import annotations

import torch
from torch import nn


def initialize_teacher_from_student(student: nn.Module, teacher: nn.Module) -> None:
    teacher.load_state_dict(student.state_dict())
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


def _student_key(student_dict: dict[str, torch.Tensor], name: str) -> str | None:
    """Resolve teacher param/buffer name to student key (handles DDP 'module.' prefix)."""
    if name in student_dict:
        return name
    ddp_name = f"module.{name}"
    if ddp_name in student_dict:
        return ddp_name
    return None


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher.named_parameters():
        key = _student_key(student_params, name)
        if key is not None:
            teacher_param.mul_(decay).add_(student_params[key].detach(), alpha=1.0 - decay)

    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        key = _student_key(student_buffers, name)
        if key is not None:
            teacher_buffer.copy_(student_buffers[key])
