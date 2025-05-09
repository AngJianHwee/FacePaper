import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMobileNetShallow_s_two_task(nn.Module):
    def __init__(self, input_channels, n_classes_task1, n_classes_task2, input_size=224, use_attention=False, attention_channels=64):
        super(AttentionMobileNetShallow_s_two_task, self).__init__()
        self.input_channels = input_channels
        self.n_classes_task1 = n_classes_task1
        self.n_classes_task2 = n_classes_task2
        self.input_size = input_size
        self.use_attention = use_attention
        self.attention_channels = attention_channels

        # Attention layers (only used if use_attention=True)
        if self.use_attention:
            self.norm = nn.LayerNorm(self.attention_channels)
            self.mha = nn.MultiheadAttention(embed_dim=self.attention_channels, num_heads=1, batch_first=True)
            self.scale = nn.Parameter(torch.zeros(1))
            # Initial conv to transform input channels to attention_channels
            self.att_conv = nn.Conv2d(input_channels, self.attention_channels, 1, 1, 0, bias=False)

        # Helper function for standard convolution with batch norm and ReLU
        def conv_batch_norm(input_channels, output_channels, stride):
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)   
            )   

        # Helper function for depthwise separable convolution
        def conv_depth_wise(input_channels, output_channels, stride):
            return nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 3, stride, 1, groups=input_channels, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        if self.input_size == 224:
            self.shared_conv = nn.Sequential(
                conv_batch_norm(self.attention_channels if self.use_attention else input_channels, 32, 2),
                conv_depth_wise(32, 64, 1),
                conv_depth_wise(64, 128, 2),
                conv_depth_wise(128, 128, 1),
                conv_depth_wise(128, 256, 2),
                conv_depth_wise(256, 256, 1),
                conv_depth_wise(256, 512, 2),
            )
        elif self.input_size == 32:
            self.shared_conv = nn.Sequential(
                conv_batch_norm(self.attention_channels if self.use_attention else input_channels, 32, 1),
                conv_depth_wise(32, 64, 1),
                conv_depth_wise(64, 128, 2),
                conv_depth_wise(128, 256, 1),
                conv_depth_wise(256, 512, 2),
            )
        else:
            raise ValueError("Input size must be either 32 or 224")

        # Task-specific feature extractors
        self.task1_conv = nn.Sequential(
            conv_depth_wise(512, 512, 1),
            conv_depth_wise(512, 1024, 2),
            conv_depth_wise(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.task2_conv = nn.Sequential(
            conv_depth_wise(512, 512, 1),
            conv_depth_wise(512, 1024, 2),
            conv_depth_wise(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        # Task-specific fully connected layers
        self.fc1 = nn.Linear(1024, n_classes_task1)
        self.fc2 = nn.Linear(1024, n_classes_task2)

    def apply_attention(self, x):
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h * w).transpose(1, 2)  # BSxHWxC
        x_att = self.norm(x_att)
        att_out, att_map = self.mha(x_att, x_att, x_att)
        return att_out.transpose(1, 2).reshape(bs, c, h, w), att_map

    def forward(self, x, return_att_map=False, return_latent=False):
        if self.use_attention:
            x = self.att_conv(x)
            x_att, att_map = self.apply_attention(x)
            x = x + self.scale * x_att  # Residual connection
        else:
            att_map = None
            x_att = None

        # Shared feature extraction
        x = self.shared_conv(x)

        # Task-specific feature extraction
        x1 = self.task1_conv(x)
        x2 = self.task2_conv(x)

        # Reshape and apply task-specific FC layers
        x1 = x1.view(-1, 1024)
        x2 = x2.view(-1, 1024)

        out1 = self.fc1(x1)
        out2 = self.fc2(x2)

        if return_latent:
            latent = x.clone()

        if return_att_map:
            if return_latent:
                return (out1, out2), att_map, x_att, latent
            else:
                return (out1, out2), att_map, x_att
        else:
            if return_latent:
                return (out1, out2), latent
            else:
                return (out1, out2)

