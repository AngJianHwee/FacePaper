# 这是一个神经网络模型文件，名字叫 AttentionMobileNetShallow_s_three_task_IAT。
# 简单来说，它是一个用来处理图像的“大脑”，特别之处在于它有“注意力”机制，
# 并且可以同时完成三个不同的任务（比如识别图像中的三种不同东西）。
# 它的设计灵感来源于MobileNet，所以它比较“轻量级”，适合在手机或嵌入式设备上运行。

import torch # 导入PyTorch库，这是搭建神经网络的工具。
import torch.nn as nn # 导入神经网络模块，比如各种层（卷积层、全连接层等）。
import torch.nn.functional as F # 导入神经网络的函数，比如激活函数、池化函数等。
from torch.autograd import Function # 导入Function，用于自定义反向传播。

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
    # input_channels: 输入图像的通道数，比如彩色图像是3（红绿蓝）。
    # n_classes_task1, n_classes_task2, n_classes_task3: 三个任务各自的分类数量。
    # input_size: 输入图像的大小，比如224x224像素。
    # use_attention: 是否使用注意力机制，True表示用，False表示不用。
    # attention_channels: 注意力机制的通道数。
    # grad_reverse: 梯度反转的强度，如果不是0，就会启用身份对抗学习。
    # num_subjects: 要识别的人脸身份的数量。
    def __init__(self, input_channels, n_classes_task1, n_classes_task2, n_classes_task3, input_size=224, use_attention=False, attention_channels=64, grad_reverse=0, num_subjects=0):
        # 调用父类nn.Module的初始化方法，这是必须的。
        super(AttentionMobileNetShallow_s_three_task_IAT, self).__init__()
        # 把传入的参数保存起来，方便后面使用。
        self.input_channels = input_channels
        self.n_classes_task1 = n_classes_task1
        self.n_classes_task2 = n_classes_task2
        self.n_classes_task3 = n_classes_task3
        self.input_size = input_size
        self.use_attention = use_attention
        self.attention_channels = attention_channels
        self.grad_reverse = grad_reverse # 梯度反转的强度，默认是0（不启用）。
        self.num_subjects = num_subjects # 身份识别的数量。

        # 注意力层（只有当use_attention为True时才使用）。
        # 注意力层就像给模型加了一双“眼睛”，让它知道图像的哪些部分更重要。
        if self.use_attention:
            # 归一化层，让数据更稳定，方便学习。
            self.norm = nn.LayerNorm(self.attention_channels)
            # 多头注意力机制，可以同时从不同角度关注图像信息。
            # embed_dim: 输入特征的维度。
            # num_heads: 注意力头的数量，这里是1个。
            # batch_first: 输入数据的批次维度是否在最前面。
            self.mha = nn.MultiheadAttention(embed_dim=self.attention_channels, num_heads=1, batch_first=True)
            # 一个可学习的缩放参数，用来调整注意力输出的强度。
            self.scale = nn.Parameter(torch.zeros(1))
            # 一个初始的卷积层，用来把输入图像的通道数转换成注意力机制需要的通道数。
            self.att_conv = nn.Conv2d(input_channels, self.attention_channels, 1, 1, 0, bias=False)

        # 这是一个辅助函数，用来创建标准的卷积层块。
        # 包含：卷积层（Conv2d）、批归一化（BatchNorm2d）和ReLU激活函数。
        # 卷积层：提取图像特征。
        # 批归一化：让训练更稳定。
        # ReLU：增加非线性，让模型能学习更复杂的模式。
        def conv_batch_norm(input_channels, output_channels, stride):
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)   
            )   

        # 这是一个辅助函数，用来创建深度可分离卷积层块。
        # 深度可分离卷积是一种特殊的卷积，可以大大减少计算量，让模型更轻量。
        # 它分成两步：深度卷积（对每个通道单独卷积）和逐点卷积（1x1卷积，混合通道信息）。
        def conv_depth_wise(input_channels, output_channels, stride):
            return nn.Sequential(
                # 深度卷积：对每个输入通道单独进行卷积。
                nn.Conv2d(input_channels, input_channels, 3, stride, 1, groups=input_channels, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(inplace=True),
                # 逐点卷积：用1x1的卷积核混合不同通道的信息。
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        # 根据输入图像的大小，构建共享的特征提取器。
        # 共享特征提取器就像模型的基础“骨架”，所有任务都会用到它提取的特征。
        if self.input_size == 224:
            self.shared_conv = nn.Sequential(
                # 如果使用了注意力，输入通道就是attention_channels，否则是input_channels。
                conv_batch_norm(self.attention_channels if self.use_attention else input_channels, 32, 2),
                conv_depth_wise(32, 64, 1),
                conv_depth_wise(64, 128, 2),
                conv_depth_wise(128, 128, 1),
                conv_depth_wise(128, 256, 2),
                conv_depth_wise(256, 256, 1),
                conv_depth_wise(256, 512, 2),
            )
        # 如果输入图像大小是32x32。
        elif self.input_size == 32:
            self.shared_conv = nn.Sequential(
                conv_batch_norm(self.attention_channels if self.use_attention else input_channels, 32, 1),
                conv_depth_wise(32, 64, 1),
                conv_depth_wise(64, 128, 2),
                conv_depth_wise(128, 256, 1),
                conv_depth_wise(256, 512, 2),
            )
        # 如果输入大小不是224也不是32，就报错。
        else:
            raise ValueError("Input size must be either 32 or 224")

        # 任务1的特征提取器。
        # 在共享特征的基础上，为任务1提取更具体的特征。
        self.task1_conv = nn.Sequential(
            conv_depth_wise(512, 512, 1),
            conv_depth_wise(512, 1024, 2),
            conv_depth_wise(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1) # 自适应平均池化，把特征图大小变成1x1，方便接到全连接层。
        )

        # 任务2的特征提取器，和任务1类似。
        self.task2_conv = nn.Sequential(
            conv_depth_wise(512, 512, 1),
            conv_depth_wise(512, 1024, 2),
            conv_depth_wise(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        # 任务3的特征提取器，和任务1类似。
        self.task3_conv = nn.Sequential(
            conv_depth_wise(512, 512, 1),
            conv_depth_wise(512, 1024, 2),
            conv_depth_wise(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        # 任务1的全连接层（分类头）。
        # 就像一个“决策器”，根据任务1提取的特征进行最终分类。
        self.fc1 = nn.Linear(1024, n_classes_task1)
        # 任务2的全连接层。
        self.fc2 = nn.Linear(1024, n_classes_task2)
        # 任务3的全连接层。
        self.fc3 = nn.Linear(1024, n_classes_task3)

        # 如果启用了梯度反转（身份对抗学习），就添加一个用于身份识别的分类头。
        # 这个ID_head就是用来识别是谁的脸的。
        if not self.grad_reverse == 0:
            self.ID_head = nn.Linear(1024, num_subjects) # 身份识别的全连接层。

    # 应用注意力机制的函数。
    def apply_attention(self, x):
        # 获取输入x的批次大小、通道数、高度和宽度。
        bs, c, h, w = x.shape
        # 调整x的形状，以便多头注意力机制处理。
        # 把它变成 (批次大小, 高*宽, 通道数) 的形状。
        x_att = x.reshape(bs, c, h * w).transpose(1, 2)  # BSxHWxC
        # 对注意力输入进行归一化。
        x_att = self.norm(x_att)
        # 通过多头注意力机制处理，得到注意力输出和注意力图。
        # 注意力图可以告诉我们模型在图像的哪些区域更“关注”。
        att_out, att_map = self.mha(x_att, x_att, x_att)
        # 把注意力输出的形状变回 (批次大小, 通道数, 高, 宽)。
        return att_out.transpose(1, 2).reshape(bs, c, h, w), att_map

    # forward方法定义了数据从输入到输出的完整流程。
    # x: 输入图像数据。
    # return_att_map: 是否返回注意力图。
    # return_latent: 是否返回中间的潜在特征。
    def forward(self, x, return_att_map=False, return_latent=False):
        # 如果使用了注意力机制。
        if self.use_attention:
            # 先通过att_conv转换通道数。
            x = self.att_conv(x)
            # 然后应用注意力机制。
            x_att, att_map = self.apply_attention(x)
            # 将注意力输出加回到原始特征上（残差连接），并用scale参数调整强度。
            # 残差连接可以帮助模型更好地学习。
            x = x + self.scale * x_att  # Residual connection
        # 如果没有使用注意力机制。
        else:
            att_map = None # 注意力图为空。
            x_att = None # 注意力输出为空。

        # 共享特征提取：数据通过共享的卷积层提取特征。
        x = self.shared_conv(x)

        # 任务特定的特征提取：将共享特征分别输入到三个任务各自的特征提取器。
        x1 = self.task1_conv(x)
        x2 = self.task2_conv(x)
        x3 = self.task3_conv(x)

        # 调整特征的形状，以便全连接层处理。
        # -1表示自动计算批次大小，1024是特征维度。
        x1 = x1.view(-1, 1024)
        x2 = x2.view(-1, 1024)
        x3 = x3.view(-1, 1024)

        # 通过任务特定的全连接层进行最终分类预测。
        out1 = self.fc1(x1)
        out2 = self.fc2(x2)
        out3 = self.fc3(x3)

        # 如果需要返回潜在特征。
        if return_latent:
            latent = x.clone() # 复制一份共享特征作为潜在特征。

        # 如果启用了梯度反转（身份对抗学习）。
        if not self.grad_reverse == 0:
            # 对共享特征应用梯度反转，让模型在识别身份时“犯错”。
            x_id = GradReverse.apply(x.view(-1, 1024), self.grad_reverse)
            # 用ID分类头预测人脸身份。
            ID_pred = self.ID_head(x_id)
            # 根据参数决定返回哪些结果。
            if return_att_map:
                if return_latent:
                    # 返回三个任务的预测结果、身份预测、注意力图、注意力输出和潜在特征。
                    return (out1, out2, out3, ID_pred), att_map, x_att, latent
                else:
                    # 返回三个任务的预测结果、身份预测、注意力图和注意力输出。
                    return (out1, out2, out3, ID_pred), att_map, x_att
            else:
                if return_latent:
                    # 返回三个任务的预测结果、身份预测和潜在特征。
                    return (out1, out2, out3, ID_pred), latent
                else:
                    # 返回三个任务的预测结果和身份预测。
                    return (out1, out2, out3, ID_pred)
        # 如果没有启用梯度反转。
        else:
            # 根据参数决定返回哪些结果。
            if return_att_map:
                if return_latent:
                    # 返回三个任务的预测结果、注意力图、注意力输出和潜在特征。
                    return (out1, out2, out3), att_map, x_att, latent
                else:
                    # 返回三个任务的预测结果、注意力图和注意力输出。
                    return (out1, out2, out3), att_map, x_att
            else:
                if return_latent:
                    # 返回三个任务的预测结果和潜在特征。
                    return (out1, out2, out3), latent
                else:
                    # 只返回三个任务的预测结果。
                    return (out1, out2, out3)
