import torch.nn as nn
import torch

# class GRN(nn.Module):
#     """ GRN (Global Response Normalization) layer
#     """
#
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
#
#     def forward(self, x):
#         Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
#         Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
#         return self.gamma * (x * Nx) + self.beta + x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.silu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.silu = nn.SiLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.silu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.silu(out)

        return out

class GuideResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(GuideResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 128
        self.in_channel2 = 128

        self.groups = groups
        self.width_per_group = width_per_group

        # Backbone for x1
        self.conv1_x1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_x1 = nn.BatchNorm2d(self.in_channel)
        self.silu = nn.SiLU(inplace=True)

        self.downconv_x1 = nn.Conv2d(self.in_channel, self.in_channel, kernel_size=3, stride=2, padding=1)
        self.bn1_downconv_x1 = nn.BatchNorm2d(self.in_channel)
        # self.maxpool_x1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # Backbone for x2
        # self.conv1_x2 = nn.Conv2d(3, self.in_channel2 // 2, kernel_size=3, stride=2,
        #                           padding=1, bias=False)
        self.conv1_x2 = nn.Conv2d(3, self.in_channel2, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_x2 = nn.BatchNorm2d(self.in_channel2)

        # self.downconv_x2 = nn.Conv2d(self.in_channel // 2, self.in_channel // 2, kernel_size=3, stride=2, padding=1)
        # self.bn1_downconv_x2 = nn.BatchNorm2d(self.in_channel // 2)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)


        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # self.fc = nn.Linear(512 * block.expansion * 2, num_classes)  # Concatenated feature size is 4 times
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # Concatenated feature size is 4 times

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def _random_select_features(self, feature_map):
        """ 随机选择特征图，如果特征图数量超过64，则保留64幅 """
        if feature_map.size(1) > 64:  # 检查通道数
            indices = torch.randperm(feature_map.size(1))[:64]  # 随机选择64个索引
            return feature_map[:, indices, :, :]  # 保留随机选择的特征图
        return feature_map  # 如果不超过64，则返回原特征图

    def forward(self, x1, x2, return_features=False):
        features = []
        if return_features:
            # Path for x1
            x1 = self.conv1_x1(x1)
            x1 = self.bn1_x1(x1)
            x1 = self.silu(x1)
            # x1 = self.maxpool_x1(x1)
            x1 = self.downconv_x1(x1)
            x1 = self.bn1_downconv_x1(x1)
            x1 = self.silu(x1)
            features.append(self._random_select_features(x1))

            # Path for x2
            x2 = self.conv1_x2(x2)
            x2 = self.bn1_x2(x2)
            x2 = self.silu(x2)
            features.append(self._random_select_features(x2))

            # Fusion x1 and x2
            # x = torch.cat((x1, x2), dim=1)
            x = torch.add(x1, x2)

            features.append(self._random_select_features(x))

            x = self.layer1(x)
            features.append(self._random_select_features(x))
            x = self.layer2(x)
            features.append(self._random_select_features(x))
            x = self.layer3(x)
            features.append(self._random_select_features(x))
            x = self.layer4(x)
            features.append(self._random_select_features(x))

            if self.include_top:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
            return x, features  # 返回分类结果和特征图
        else:
            # x = torch.cat((x1, x2), dim=1)
            # Path for x1
            x1 = self.conv1_x1(x1)
            x1 = self.bn1_x1(x1)
            x1 = self.silu(x1)

            # x1 = self.maxpool_x1(x1)
            x1 = self.downconv_x1(x1)
            x1 = self.bn1_downconv_x1(x1)
            x1 = self.silu(x1)

            # Path for x2
            x2 = self.conv1_x2(x2)
            x2 = self.bn1_x2(x2)
            x2 = self.silu(x2)

            # x2 = self.downconv_x2(x2)
            # x2 = self.bn1_downconv_x2(x2)
            # x2 = self.silu(x2)

            # Fusion x1 and x2
            # x = torch.cat((x1, x2), dim=1)
            x = torch.add(x1, x2)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            if self.include_top:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
        return x


def GuideResNet34(num_classes=1000, include_top=True):
    return GuideResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def GuideResNet50(num_classes=1000, include_top=True):
    return GuideResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def GuideResNet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return GuideResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)