import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Any
from resnet import ResNet, BasicBlock
from config import *
from torch.optim.lr_scheduler import StepLR

NUM_CLASSES = 10  

# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 무작위로 크롭하여 데이터 다양화
    transforms.RandomHorizontalFlip(),    # 무작위로 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
])


# CIFAR-10 데이터셋 로드
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resnet 18 선언하기
## TODO
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES).to(device)


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
###
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # L2 정규화 추가
scheduler = StepLR(optimizer, step_size=30, gamma=0.1) ##스케쥴러 생성성

# 학습 
def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
    model.train()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        ##
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        ##
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy: float = 100. * correct / total
    print(f"Train Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")

# 평가 
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> None:
    model.eval()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy: float = 100. * correct / total
    print(f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")

# 학습 및 평가 루프
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train(model, train_loader, criterion, optimizer, device)
    evaluate(model, test_loader, criterion, device)
    scheduler.step() ###
# 모델 저장
torch.save(model.state_dict(), "resnet18_checkpoint.pth")
print(f"Model saved to resnet18_checkpoint.pth")
