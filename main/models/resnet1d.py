import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """基础残差块（2个1D卷积层）"""
    expansion = 1  # 输出通道数扩展倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 卷积步长（用于下采样）
            downsample: 下采样模块（用于调整跳跃连接的维度）
        """
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)  # 批归一化
        self.relu = nn.ReLU(inplace=True)        # ReLU激活
        
        # 第二个卷积层（步长固定为1，不改变序列长度）
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample  # 跳跃连接的下采样模块
        self.stride = stride

    def forward(self, x):
        residual = x  # 保存输入用于跳跃连接

        # 主路径计算
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 跳跃连接（若需要下采样则先调整维度）
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差相加
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """瓶颈残差块（1x1-3x1-1x1卷积，减少计算量）"""
    expansion = 2  # 输出通道数扩展倍数（最后一个1x1卷积将通道数扩大2倍）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1卷积（降维）
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 3x1卷积（核心卷积）
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 1x1卷积（升维）
        self.conv3 = nn.Conv1d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """1D ResNet主类"""
    def __init__(self, block, layers, input_channels=1):
        """
        Args:
            block: 残差块类型（BasicBlock或Bottleneck）
            layers: 每个阶段的残差块数量（如[2,2,2,2]对应ResNet-18）
            input_channels: 输入数据的通道数（如音频可能为1，多通道时间序列可能为n）
            num_classes: 分类任务的类别数
        """
        super(ResNet1D, self).__init__()
        self.in_channels = 64  # 初始卷积层输出通道数
        # 初始卷积层（将输入映射到64通道）
        self.conv1 = nn.Conv1d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 最大池化层（进一步下采样）
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差块阶段（4个阶段）
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        
        # # 全局平均池化（将序列长度降为1）
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # # 全连接层（输出分类结果）
        # self.fc = nn.Linear(512 * block.expansion, output_channels)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """构建一个残差块阶段"""
        downsample = None
        # 若步长不为1或输入输出通道数不匹配，需要下采样调整维度
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        # 第一个残差块（可能包含下采样）
        layers.append(block(
            self.in_channels, out_channels, stride, downsample
        ))
        self.in_channels = out_channels * block.expansion
        
        # 后续残差块（步长为1，不改变维度）
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        # 输入形状: (batch_size, input_channels, seq_len)
        
        x = self.conv1(x)       # 卷积: (batch, 64, seq_len/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 池化: (batch, 64, seq_len/4)

        x = self.layer1(x)      # 阶段1: 保持通道数64*expansion
        x = self.layer2(x)      # 阶段2: 通道数128*expansion，长度减半
        x = self.layer3(x)      # 阶段3: 通道数256*expansion，长度减半
        x = self.layer4(x)      # 阶段4: 通道数512*expansion，长度减半

        # x = self.avgpool(x)     # 全局池化: (batch, 512*expansion, 1)
        # x = torch.flatten(x, 1) # 展平: (batch, 512*expansion)
        # x = self.fc(x)          # 全连接: (batch, num_classes)

        return x


# 预定义常用的1D ResNet模型
def resnet18_1d(input_channels=1):
    """1D ResNet-18（使用BasicBlock）"""
    return ResNet1D(BasicBlock, [2, 2, 2, 2], input_channels)


def resnet34_1d(input_channels=1):
    """1D ResNet-34（使用BasicBlock）"""
    return ResNet1D(BasicBlock, [3, 4, 6, 3], input_channels)


def resnet50_1d(input_channels=1):
    """1D ResNet-50（使用Bottleneck）"""
    return ResNet1D(Bottleneck, [3, 4, 6, 3], input_channels)


# 测试代码
if __name__ == "__main__":
    # 生成随机1D序列数据 (batch_size=8, channels=1, seq_len=1000)
    x = torch.randn(8, 1, 1000)
    
    # 初始化模型
    model = resnet18_1d(input_channels=1, num_classes=10)
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应输出 (8, 10)