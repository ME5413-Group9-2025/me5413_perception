import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from MyNet import CustomResNet18

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28, 28)),
])

model = CustomResNet18()
model.load_state_dict(torch.load('mnist_resnet18.pth'))
model.eval()


def recognize(image: np.ndarray):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, number = torch.max(output, 1)
    return image, number


if __name__ == '__main__':
    for i in range(10):
        image_path = f'mnist_test_images\\{i:05d}.jpg'
        image = Image.open(image_path)
        image = 255 - np.array(image)
        image, predicted = recognize(image, model)

        image = np.array(image.squeeze(0))
        image = np.reshape(image, (28, 28))
        plt.imshow(image, cmap='gray')
        plt.text(5, 1, f'Predicted: {predicted.item()}', color='red', fontsize=16, ha='center')
        plt.axis('off')
        plt.show()
