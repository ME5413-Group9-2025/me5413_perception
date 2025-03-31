import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from MyNet import CustomResNet18

# 加载自定义的ResNet18模型
model = CustomResNet18()
model.load_state_dict(torch.load('mnist_resnet18.pth'))
model.eval()

def predict(image, model):
    image = image.convert('L')
    image = 255 - np.array(image)
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)
    # 进行模型预测
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return image, predicted

# 读取前十张图片进行预测
for i in range(10):
    image_path = f'mnist_test_images\\{i:05d}.jpg'
    image = Image.open(image_path)
    # image = image.convert('L')  # 转换为灰度图
    # # print(image.size)
    # # plt.imshow(image, cmap='gray')
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.5,), (0.5,))
    # ])
    # image = transform(image).unsqueeze(0)
    #
    # # 进行模型预测
    # with torch.no_grad():
    #     output = model(image)
    #     _, predicted = torch.max(output, 1)
    #
    # # 在原图片右上角显示预测结果
    image, predicted = predict(image, model)


    image = np.array(image.squeeze(0))  # 将图像数据转换为numpy数组，并反归一化
    image = np.reshape(image,(28,28))
    plt.imshow(image, cmap='gray')
    plt.text(5, 1, f'Predicted: {predicted.item()}', color='red', fontsize=16, ha='center')
    plt.axis('off')
    plt.show()
