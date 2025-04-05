import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from .MyNet import CustomResNet18
import copy


model = CustomResNet18()
model.load_state_dict(torch.load('minist/mnist_resnet18.pth'))
model.eval()


# image_path = "../image.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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

def recognize(image, detection_threshold=None, area_threshold=100.0, kernel_size=3, gs_size = 5, return_max=False):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value = 80
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # if image size changed, change here
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filtered_image = cv2.erode(binary_image, kernel, iterations=1)

    
    contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    vis_image = copy.deepcopy(image)
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    max_area = 0
    max_result = [0,(0,0,0,0)]
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

        # if image size changed, change here
        blurred = cv2.GaussianBlur(morph_close, (gs_size , gs_size ), 0)
        _, threshold_blur = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

        
        final_image = np.ones((28, 28), dtype=np.uint8) * 255
        start_x = (28 - 14) // 2
        start_y = (28 - 21) // 2
        final_image[start_y:start_y + 21, start_x:start_x + 14] = threshold_blur

        
        processed_image, predicted = predict(final_image, model)

       
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)

        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        results.append([predicted.item(), (x, y, x+w, y+h)])
        if area > max_area:
            max_area = area
            max_result = [predicted.item(), (x, y, x+w, y+h)]
    print(results)
    cv2.imshow("image", vis_image)
    cv2.imwrite("image.png", image)
    cv2.waitKey(0)
    if return_max:
        return max_result
    else:
        return results


if __name__ == "__main__":
    results = recognize(image)
    print(results)
