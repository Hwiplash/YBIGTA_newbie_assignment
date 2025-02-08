import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
from typing import Type, List, Optional

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()

        self.conv1: nn.Conv2d = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(planes)

        self.conv2: nn.Conv2d = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        
        self.shortcut: nn.Sequential = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        ## TODO
        # 첫 번째 합성곱 + 배치 정규화 + ReLU
        out = F.relu(self.bn1(self.conv1(x)))

        # 두 번째 합성곱 + 배치 정규화
        out = self.bn2(self.conv2(out))

        # 스킵 연결 추가
        out += self.shortcut(x)

        # 최종 ReLU 적용
        out = F.relu(out)
        
        ###
        return out
        
    
class ResNet(nn.Module):
    def __init__(self, block: Type[BasicBlock], num_blocks: List[int], num_classes: int = 2, init_weights: bool = True) -> None:
        super().__init__()

        self.in_channels: int = 64

        ## TODO
        # Resnet layer를 구현하세요!
        # Hint: 두번째 layer부터는 _make_layer 메서드를 활용하세요! 
        # conv1: 입력 채널 3 → 출력 채널 64, kernel_size=7, stride=2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers (conv2_x, conv3_x, conv4_x, conv5_x)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # conv3_x
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # conv4_x
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # conv5_x
        ####

        self.avg_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc: nn.Linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:

        strides: List[int] = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        
        ## TODO
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion  # 다음 블록의 입력 채널 업데이트
       ####
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## TODO
        # conv1 → BatchNorm → ReLU → MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Average Pooling → Flatten → Fully Connected Layer
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)

        return x