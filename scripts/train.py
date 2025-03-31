import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from MyNet import CustomResNet18, CustomImageDataset
import os

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    # 你可以根据需要添加其他预处理，例如归一化
])

# 指定保存图像的目录和标签文件的路径
images_dir = './pro_mnist_train_images'
l_dir = './mnist_train_images'
labels_file = os.path.join(l_dir, 'labels.txt')

# 创建自定义数据集实例
trainset = CustomImageDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)

# 使用DataLoader加载数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# 创建自定义ResNet-18模型实例
custom_resnet18 = CustomResNet18()

# 使用GPU进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
custom_resnet18.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(custom_resnet18.parameters(), lr=0.001, momentum=0.9)
# 如果你更喜欢Adam优化器，可以选择以下代码：
# optimizer = torch.optim.Adam(custom_resnet18.parameters(), lr=0.001)

# 训练模型
for epoch in range(5):  # 遍历数据集5次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        # 前向传播
        outputs = custom_resnet18(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:  # 每200个batch打印一次
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

# 保存训练好的模型
PATH = './mnist_resnet18.pth'
torch.save(custom_resnet18.state_dict(), PATH)
