import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMobileNetShallow_s(nn.Module):
    def __init__(self, input_channels, n_classes, input_size=224, use_attention=False, attention_channels=64):
        super(AttentionMobileNetShallow_s, self).__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.input_size = input_size
        self.use_attention = use_attention
        self.attention_channels = attention_channels  # New parameter for attention channels

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

            # Main model architecture (224x224 input)
            self.model = nn.Sequential(
                conv_batch_norm(self.attention_channels if self.use_attention else input_channels, 32, 2),
                conv_depth_wise(32, 64, 1),
                conv_depth_wise(64, 128, 2),
                conv_depth_wise(128, 128, 1),
                conv_depth_wise(128, 256, 2),
                conv_depth_wise(256, 256, 1),
                conv_depth_wise(256, 512, 2),
                # conv_depth_wise(512, 512, 1),     # Reducing the number of depthwise convolutions to speed up the model
                # conv_depth_wise(512, 512, 1),
                # conv_depth_wise(512, 512, 1),
                # conv_depth_wise(512, 512, 1),
                conv_depth_wise(512, 512, 1),
                conv_depth_wise(512, 1024, 2),
                conv_depth_wise(1024, 1024, 1),
                nn.AdaptiveAvgPool2d(1)
            )
        elif self.input_size == 32:
            # Modified model for 32x32 inputs
            self.model = nn.Sequential(
                conv_batch_norm(self.attention_channels if self.use_attention else input_channels, 32, 1),
                conv_depth_wise(32, 64, 1),
                conv_depth_wise(64, 128, 2),
                conv_depth_wise(128, 256, 1),
                conv_depth_wise(256, 512, 2),
                # conv_depth_wise(512, 512, 1), # Reducing the number of depthwise convolutions to speed up the model
                conv_depth_wise(512, 512, 1),
                conv_depth_wise(512, 1024, 2),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            raise ValueError("Input size must be either 32 or 224")
                        
        self.fc = nn.Linear(1024, n_classes)

    
    def apply_attention(self, x):
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h * w).transpose(1, 2)  # BSxHWxC
        x_att = self.norm(x_att)
        att_out, att_map = self.mha(x_att, x_att, x_att)
        return att_out.transpose(1, 2).reshape(bs, c, h, w), att_map

    def forward(self, x, return_att_map=False, return_latent=False):
        if self.use_attention:
            x = self.att_conv(x)
            # x = self.apply_attention(x)
            x_att, att_map = self.apply_attention(x)
            x = x + self.scale * x_att  # Residual connection
        else:
            att_map = None
            x_att = None
        # now pass x through the model
        
        x = self.model(x)
        
        x = x.view(-1, 1024)
        # copy x to latent
        if return_latent:
            latent = x.clone()
        x = self.fc(x)

        if return_att_map:
            if return_latent:
                return x, att_map, x_att, latent
            else:
                return x, att_map, x_att
        else:
            if return_latent:
                return x, latent
            else:
                return x

