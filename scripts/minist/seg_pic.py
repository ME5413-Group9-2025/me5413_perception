import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from MyNet import CustomResNet18


model = CustomResNet18()
model.load_state_dict(torch.load('mnist_resnet18.pth'))
model.eval()


image_path = "8.png" 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def predict(image, model):
    image = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return image, predicted

def recognize(image, area_threshold=100.0):

    threshold_value = 80
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    
    kernel = np.ones((3, 3), np.uint8)
    filtered_image = cv2.erode(binary_image, kernel, iterations=1)

    
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_threshold:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        roi = binary_image[y:y + h, x:x + w]
        occupancy_ratio = np.sum(roi == 255) / roi.size
        if occupancy_ratio > 0.8:
            continue

        
        resized_roi = cv2.resize(roi, (14, 21))

        
        inverted_image = cv2.bitwise_not(resized_roi)

        
        morph_close = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)

        
        blurred = cv2.GaussianBlur(morph_close, (5, 5), 0)
        _, threshold_blur = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

        
        final_image = np.ones((28, 28), dtype=np.uint8) * 255
        start_x = (28 - 14) // 2
        start_y = (28 - 21) // 2
        final_image[start_y:start_y + 21, start_x:start_x + 14] = threshold_blur

        
        processed_image, predicted = predict(final_image, model)

       
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)

        
        results.append([predicted.item(), (center_x, center_y)])

    return results

results = recognize(image)

print(results)
