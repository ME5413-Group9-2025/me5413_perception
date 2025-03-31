from torchvision.models import resnet18
import torch.nn as nn
import torchvision.models as models
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None):
        """
        Args:
            images_dir (string): 存储图像的文件夹路径
            labels_file (string): 存储图像与标签映射关系的标签文件路径
            transform (callable, optional): 可选的图像预处理函数
        """
        self.images_dir = images_dir
        self.transform = transform
        self.labels = []
        self.image_files = []

        # 读取标签文件
        with open(labels_file, 'r') as file:
            for line in file:
                image_file, label = line.strip().split()
                self.image_files.append(image_file)
                self.labels.append(int(label))  # 将标签转换为整数

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path)
        label = self.labels[idx]

        # 如果有预处理函数，则应用该函数
        if self.transform:
            image = self.transform(image)

        return image, label


# 创建自定义的ResNet-18模型类
class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数为1
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10)  # 修改输出大小为10（适应10个类别）

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet18.fc(x)

        return x

# # 创建自定义ResNet-18模型实例
# custom_resnet18 = CustomResNet18()
