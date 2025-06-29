# 这是一个神经网络模型文件，名字叫 AttentionMobileNetShallow_s_three_task_IAT。
# 简单来说，它是一个用来处理图像的“大脑”，特别之处在于它有“注意力”机制，
# 并且可以同时完成三个不同的任务（比如识别图像中的三种不同东西）。
# 它的设计灵感来源于MobileNet，所以它比较“轻量级”，适合在手机或嵌入式设备上运行。

import torch # 导入PyTorch库，这是搭建神经网络的工具。
import torch.nn as nn # 导入神经网络模块，比如各种层（卷积层、全连接层等）。
import torch.nn.functional as F # 导入神经网络的函数，比如激活函数、池化函数等。
from torch.autograd import Function # 导入Function，用于自定义反向传播。
from .AttentionMobileNetShallow_s_three_task import AttentionMobileNetShallow_s_three_task

# 这是一个特殊的函数，叫做梯度反转层。
# 想象一下，我们想让神经网络在学习某个东西（比如识别表情）的时候，
# 故意“忘记”另一个东西（比如识别是谁的脸）。
# 这个层就是用来实现这个目的的，它在反向传播（学习过程）的时候，把梯度方向反过来。
class GradReverse(Function):
    # forward方法定义了数据正向通过这个层时会发生什么。
    # 这里它只是把输入x原样返回，但会记住一个叫lambd的参数。
    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    # backward方法定义了在反向传播（学习）时会发生什么。
    # 它会把传回来的梯度乘以-lambd，这样就实现了梯度的反转。
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None

# 定义一个神经网络类，名字叫 AttentionMobileNetShallow_s_three_task_IAT。
# 它是基于nn.Module的，nn.Module是PyTorch里所有神经网络模块的基类。
class AttentionMobileNetShallow_s_three_task_IAT(nn.Module):
    # 模型的初始化函数，就像给模型搭骨架，设置好各种零件。
    def __init__(self, existing_model: AttentionMobileNetShallow_s_three_task, grad_reverse=0.0, num_subjects=0):
        # 调用父类nn.Module的初始化方法，这是必须的。
        super(AttentionMobileNetShallow_s_three_task_IAT, self).__init__()
        self.base_model = existing_model
        self.grad_reverse = grad_reverse # 梯度反转的强度，默认是0（不启用）。
        self.num_subjects = num_subjects # 身份识别的数量。

        # 如果启用了梯度反转（身份对抗学习），就添加一个用于身份识别的分类头。
        # 这个ID_head就是用来识别是谁的脸的。
        if not self.grad_reverse == 0:
            self.ID_head = nn.Linear(1024, num_subjects) # 身份识别的全连接层。
        else:
            self.ID_head = None

    # forward方法定义了数据从输入到输出的完整流程。
    # x: 输入图像数据。
    # return_att_map: 是否返回注意力图。
    # return_latent: 是否返回中间的潜在特征。
    def forward(self, x, return_att_map=False, return_latent=False):
        # Get outputs and latent features from the base model
        # The base model's forward method already handles attention and returns latent if requested
        base_outputs = self.base_model(x, return_att_map=True, return_latent=True)
        # print shape
        for key, value in base_outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} shape: {value.shape}")
            else:
                print(f"{key} type: {type(value)}")


        # Unpack base_outputs based on what base_model returns
        # base_model returns (out1, out2, out3), att_map, x_att, latent
        (out1, out2, out3), att_map, x_att, latent = base_outputs

        print(f"out1 shape: {out1.shape}, out2 shape: {out2.shape}, out3 shape: {out3.shape}, att_map shape: {att_map.shape if return_att_map else 'N/A'}, x_att shape: {x_att.shape if return_att_map else 'N/A'}, latent shape: {latent.shape if return_latent else 'N/A'}")

        ID_pred = None
        if not self.grad_reverse == 0 and self.ID_head is not None:
            # Apply gradient reversal to the latent features from the base model
            x_id = GradReverse.apply(latent.view(-1, 1024), self.grad_reverse)
            ID_pred = self.ID_head(x_id)

        # Construct the return tuple based on original IAT model's return logic
        if not self.grad_reverse == 0:
            # Original IAT returns (out1, out2, out3, ID_pred)
            task_outputs = (out1, out2, out3, ID_pred)
        else:
            task_outputs = (out1, out2, out3)

        if return_att_map:
            if return_latent:
                return task_outputs, att_map, x_att, latent
            else:
                return task_outputs, att_map, x_att
        else:
            if return_latent:
                return task_outputs, latent
            else:
                return task_outputs
