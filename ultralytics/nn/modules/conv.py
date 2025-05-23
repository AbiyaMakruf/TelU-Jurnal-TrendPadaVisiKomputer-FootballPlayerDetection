# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/ghostnet
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (tuple): Tuple containing:
                - Equivalent kernel (torch.Tensor)
                - Equivalent bias (torch.Tensor)
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            (tuple): Tuple containing:
                - Fused kernel (torch.Tensor)
                - Fused bias (torch.Tensor)
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Select and return a particular index from input.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]

class SimAM(nn.Module):
    """
    SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks.
    https://arxiv.org/pdf/2101.08165.pdf
    """
    def __init__(self, c1, c2, e_lambda=1e-4):
        """
        Initialize SimAM module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels. Must be equal to c1 for SimAM.
            e_lambda (float): Regularization parameter lambda.
        """
        super().__init__()
        # SimAM does not change the number of channels. Assert this condition.
        assert c1 == c2, f"Input channels ({c1}) and output channels ({c2}) must be equal for SimAM."

        self.e_lambda = e_lambda
        self.activation = nn.Sigmoid() # Internal activation for the attention map

        # SimAM is parameter-free, so no learnable weights like Conv or BN are defined here.
        # We keep c1 and c2 in the signature for consistency with how parse_model might call it,
        # but c2 isn't strictly used beyond the assertion.

    def forward(self, x):
        """
        Apply SimAM attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output tensor with attention applied (same shape as input).
        """
        # Spatial dimensions
        _, _, h, w = x.size()
        # Number of elements in spatial dimensions
        n = w * h - 1

        # Calculate mean and variance
        # mean_x = x.mean(dim=[2, 3], keepdim=True) # E[x]
        # var_x = ((x - mean_x) ** 2).mean(dim=[2, 3], keepdim=True) # Var[x]

        # More efficient calculation: E[(X - mu)^2] = E[X^2] - (E[X])^2 is less stable
        # Using the direct formula from the paper for stability:
        # D = (1/n) * sum_i^n (x_i - mu)^2
        # mu = (1/N) * sum_i^N x_i
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # Denominator: 4 * (sigma_hat^2 + lambda) where sigma_hat^2 = (1/n) * sum((x - mu)^2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        # Apply attention: x * sigmoid(E)
        return x * self.activation(y)

    def forward_fuse(self, x):
        """
        Forward pass for fused model. SimAM has no fuseable components (like BN),
        so this is identical to the standard forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.forward(x)

    def __repr__(self):
        """String representation of the module."""
        return f"{self.__class__.__name__}(lambda={self.e_lambda})"


# --- Implementation of LCBHAM ---

class LCAM(nn.Module):
    """
    Lightweight Channel Attention Module (LCAM) based on the diagram.
    Uses a shared MLP for both maxpool and avgpool features.
    """
    def __init__(self, c1, r=16): # c1 = input channels, r = reduction ratio
        super().__init__()
        c_ = c1 // r # Intermediate channels
        if c_ == 0: c_ = 1 # Ensure intermediate channels is at least 1
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0, bias=False), # Use Conv2d for MLP on features
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c1, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max = self.maxpool(x)
        x_avg = self.avgpool(x)
        x_max = self.mlp(x_max)
        x_avg = self.mlp(x_avg)
        channel_att = self.sigmoid(x_max + x_avg)
        return channel_att # Output shape: (B, C, 1, 1)

class LD_SAM(nn.Module):
    """
    Lightweight Dilated Spatial Attention Module (LD-SAM) based on the diagram.
    Note: Diagram shows k=3, not explicitly dilated, but name suggests it.
           Using standard k=3 conv as per visual representation.
    """
    def __init__(self, kernel_size=3): # Kernel size for the spatial conv
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for 'same' padding"
        # Input to conv will have 2 channels (max_pool + avg_pool)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=autopad(kernel_size), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply max and average pooling along the channel dimension
        x_max, _ = torch.max(x, dim=1, keepdim=True) # Shape: (B, 1, H, W)
        x_avg = torch.mean(x, dim=1, keepdim=True)   # Shape: (B, 1, H, W)
        # Concatenate along the channel dimension
        x_cat = torch.cat([x_max, x_avg], dim=1)     # Shape: (B, 2, H, W)
        # Apply convolution and sigmoid
        spatial_att = self.sigmoid(self.conv(x_cat)) # Shape: (B, 1, H, W)
        return spatial_att

class LCBHAM(nn.Module):
    """
    Lightweight Coordinate Attention Block with Hardswish Activation Module (LCBHAM).
    Combines an initial convolution block with LCAM and LD-SAM.
    """
    def __init__(self, c1, c2, k=3, s=2): # c1=input channels, c2=output channels for initial conv
        super().__init__()
        # Initial Convolution Block
        self.conv_block = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p=None, d=1), bias=False),
            nn.BatchNorm2d(c2),
            nn.Hardswish(inplace=True)
        )
        # Attention Modules
        self.channel_att = LCAM(c2) # Takes output channels of conv_block
        self.spatial_att = LD_SAM(kernel_size=3) # Using k=3 as shown in diagram

    def forward(self, x):
        # Apply initial convolution block
        x_conv = self.conv_block(x)

        # Apply Channel Attention (LCAM)
        channel_weights = self.channel_att(x_conv)
        x_channel_refined = x_conv * channel_weights # Element-wise multiplication

        # Apply Spatial Attention (LD-SAM)
        spatial_weights = self.spatial_att(x_channel_refined)
        x_spatial_refined = x_channel_refined * spatial_weights # Element-wise multiplication

        # Output feature
        return x_spatial_refined

# --- Implementation of LCBHAM ---

class LCAM(nn.Module):
    """
    Lightweight Channel Attention Module (LCAM) based on the diagram.
    Uses a shared MLP for both maxpool and avgpool features.
    """
    def __init__(self, c1, r=16): # c1 = input channels, r = reduction ratio
        super().__init__()
        c_ = c1 // r # Intermediate channels
        if c_ == 0: c_ = 1 # Ensure intermediate channels is at least 1
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0, bias=False), # Use Conv2d for MLP on features
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c1, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max = self.maxpool(x)
        x_avg = self.avgpool(x)
        x_max = self.mlp(x_max)
        x_avg = self.mlp(x_avg)
        channel_att = self.sigmoid(x_max + x_avg)
        return channel_att # Output shape: (B, C, 1, 1)

class LD_SAM(nn.Module):
    """
    Lightweight Dilated Spatial Attention Module (LD-SAM) based on the diagram.
    Note: Diagram shows k=3, not explicitly dilated, but name suggests it.
           Using standard k=3 conv as per visual representation.
    """
    def __init__(self, kernel_size=3): # Kernel size for the spatial conv
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for 'same' padding"
        # Input to conv will have 2 channels (max_pool + avg_pool)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=autopad(kernel_size), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply max and average pooling along the channel dimension
        x_max, _ = torch.max(x, dim=1, keepdim=True) # Shape: (B, 1, H, W)
        x_avg = torch.mean(x, dim=1, keepdim=True)   # Shape: (B, 1, H, W)
        # Concatenate along the channel dimension
        x_cat = torch.cat([x_max, x_avg], dim=1)     # Shape: (B, 2, H, W)
        # Apply convolution and sigmoid
        spatial_att = self.sigmoid(self.conv(x_cat)) # Shape: (B, 1, H, W)
        return spatial_att

class LCBHAM(nn.Module):
    """
    Lightweight Coordinate Attention Block with Hardswish Activation Module (LCBHAM).
    Combines an initial convolution block with LCAM and LD-SAM.
    """
    def __init__(self, c1, c2, k=3, s=2): # c1=input channels, c2=output channels for initial conv
        super().__init__()
        # Initial Convolution Block
        self.conv_block = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p=None, d=1), bias=False),
            nn.BatchNorm2d(c2),
            nn.Hardswish(inplace=True)
        )
        # Attention Modules
        self.channel_att = LCAM(c2) # Takes output channels of conv_block
        self.spatial_att = LD_SAM(kernel_size=3) # Using k=3 as shown in diagram

    def forward(self, x):
        # Apply initial convolution block
        x_conv = self.conv_block(x)

        # Apply Channel Attention (LCAM)
        channel_weights = self.channel_att(x_conv)
        x_channel_refined = x_conv * channel_weights # Element-wise multiplication

        # Apply Spatial Attention (LD-SAM)
        spatial_weights = self.spatial_att(x_channel_refined)
        x_spatial_refined = x_channel_refined * spatial_weights # Element-wise multiplication

        # Output feature
        return x_spatial_refined