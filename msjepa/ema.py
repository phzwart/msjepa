from __future__ import annotations

import torch
from torch import nn


def initialize_teacher_from_student(student: nn.Module, teacher: nn.Module) -> None:
    teacher.load_state_dict(student.state_dict())
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher.named_parameters():
        teacher_param.mul_(decay).add_(student_params[name].detach(), alpha=1.0 - decay)

    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        if name in student_buffers:
            teacher_buffer.copy_(student_buffers[name])
