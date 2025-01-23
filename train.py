from net import AlexNet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.init as init

# 论文提供的初始化方法不行，迭代不收敛
def paper_initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # 使用标准差为0.01、均值为0的高斯分布初始化权重
            init.normal_(m.weight, mean=0, std=0.01)
            # 常数1来初始化了网络中的第二个、第四个和第五个卷积层以及全连接层中的隐含层中的所有偏置参数
            # 这种初始化权重的方法通过向ReLU提供了正的输入，来加速前期的训练
            if m in [model.conv2[0], model.conv4[0], model.conv5[0], model.fc1[1], model.fc2[1]]:
                nn.init.constant_(m.bias, 1)
            else:
                nn.init.constant_(m.bias, 0) # 常数0来初始化剩余层中的偏置参数

# 使用何凯明提出的方法初始化可以很容易收敛
# refer 2015 《Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def train():
    # model
    model = AlexNet(num_classes=10, in_channels=3)
    #initialize_weights(model)
    init_weights(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 训练数据预处理
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(227), # 论文中的第一种数据增广法：随机水平翻转+随机裁剪
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),# 论文中的第二种数据增广法：改变RGB通道的强度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4595, 0.4402, 0.3700], std=[0.2005, 0.1984, 0.1907])
    ])
    # 测试数据预处理
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4595, 0.4402, 0.3700], std=[0.2005, 0.1984, 0.1907])
    ])

    # 加载数据集
    # 每个类别的图像存放在不同的文件夹中，可以使用ImageFolder来加载数据集
    train_dataset = datasets.ImageFolder(root='F:/深度学习数据集/imagenet-10/train/', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='F:/深度学习数据集/imagenet-10/test/', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8) #训练时需要打乱，减少过拟合
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8) #验证时不需要打乱

    # 训练参数（启发式训练）
    num_epochs = 200
    initial_lr = 0.01 # 设置初始学习率为0.01
    momentum = 0.9
    weight_decay = 0.005
    lr_decay_factor = 0.1 # 验证集上的错误率停止降低时，将学习速率*0.1（缩小十倍）。
    lr_decay_patience = 3 # 在终止前学习率衰减3次
    best_val_loss = float('inf')
    lr_decay_counter = 0

    # 优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum,weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader)) :
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= le
