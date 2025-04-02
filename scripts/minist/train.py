import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from MyNet import CustomResNet18, CustomImageDataset
import os
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
])

images_dir = './pro_mnist_train_images'
l_dir = './mnist_train_images'
labels_file = os.path.join(l_dir, 'labels.txt')

trainset = CustomImageDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

custom_resnet18 = CustomResNet18()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
custom_resnet18.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(custom_resnet18.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = custom_resnet18(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './mnist_resnet18.pth'
torch.save(custom_resnet18.state_dict(), PATH)
