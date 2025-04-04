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

# 读取图片（以灰度模式）
image_path = "8.png"  # 你的图片路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def predict(image, model):
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

def seg(image):
    # 手动设置阈值
    threshold_value = 80
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # 腐蚀操作
    kernel = np.ones((3, 3), np.uint8)
    filtered_image = cv2.erode(binary_image, kernel, iterations=1)

    # 查找所有轮廓
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    roi_index = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        roi = binary_image[y:y + h, x:x + w]
        occupancy_ratio = np.sum(roi == 255) / roi.size
        if occupancy_ratio > 0.8:
            continue

        # # --- 显示 ROI ---
        # plt.figure()
        # plt.title(f"ROI #{roi_index} (area={int(area)})")
        # plt.imshow(roi, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # roi_index += 1

        # 缩放到 14x21
        resized_roi = cv2.resize(roi, (14, 21))

        # 反转图像
        inverted_image = cv2.bitwise_not(resized_roi)

        # 闭运算
        morph_close = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)

        # 高斯模糊 + 二值化
        blurred = cv2.GaussianBlur(morph_close, (5, 5), 0)
        _, threshold_blur = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

        # 创建白底图像并居中放置
        final_image = np.ones((28, 28), dtype=np.uint8) * 255
        start_x = (28 - 14) // 2
        start_y = (28 - 21) // 2
        final_image[start_y:start_y + 21, start_x:start_x + 14] = threshold_blur

        # 预测
        processed_image, predicted = predict(final_image, model)
        results.append((processed_image, predicted))

    return results
# 处理图像
results = seg(image)

# 可视化多个预测
for idx, (image_tensor, prediction) in enumerate(results):
    image_np = np.reshape(image_tensor.numpy(), (28, 28))
    plt.subplot(1, len(results), idx + 1)
    plt.imshow(image_np, cmap='gray')
    plt.title(f'Pred: {prediction.item()}', color='red')
    plt.axis('off')

plt.show()