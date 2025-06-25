import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# === Custom Conv2D with SAME padding ===
class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self.calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias,
                        self.stride, self.padding,
                        self.dilation, self.groups)

# === Bottleneck Block ===
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2dSame(out_channels, out_channels, 3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.i_downsample = i_downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.i_downsample:
            identity = self.i_downsample(identity)
        out += identity
        return self.relu(out)

# === Full ResNet Model ===
class ResNet(nn.Module):
    def __init__(self, ResBlock, layers, num_classes=7, num_channels=3):
        super().__init__()
        self.in_channels = 64
        self.conv1 = Conv2dSame(num_channels, 64, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.layer1 = self._make_layer(ResBlock, layers[0], 64, 1)
        self.layer2 = self._make_layer(ResBlock, layers[1], 128, 2)
        self.layer3 = self._make_layer(ResBlock, layers[2], 256, 2)
        self.layer4 = self._make_layer(ResBlock, layers[3], 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512 * ResBlock.expansion, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu1 = nn.ReLU()

    def _make_layer(self, block, blocks, planes, stride):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [block(self.in_channels, planes, downsample, stride)]
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))
        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)

# === LSTM Model ===
class LSTMPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        return self.softmax(x)
