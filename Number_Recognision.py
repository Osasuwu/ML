from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Определение архитектуры нейросети
class DigitRecognitionNet(nn.Module):
    def __init__(self):
        super(DigitRecognitionNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Функция обучения
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Эпоха {epoch}, Батч {batch_idx}, Потеря: {loss.item():.6f}')

# Функция тестирования
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Тестовая потеря: {test_loss:.4f}, Точность: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

# Функция для тестирования собственных изображений
def predict_custom_image(model, device, image_path):
    # Загрузка изображения
    image = Image.open(image_path)
    
    # Преобразование в оттенки серого и изменение размера
    image = image.convert('L')  # Преобразовать в оттенки серого
    image = image.resize((28, 28))  # Изменить размер до 28x28
    
    # Конвертировать в numpy массив
    image_array = np.array(image)
    
    # ВАЖНО: Инвертировать цвета (MNIST имеет белые цифры на черном фоне)
    image_array = 255 - image_array
    
    # Преобразовать обратно в PIL Image
    image_inverted = Image.fromarray(image_array)
    
    # Применить трансформации как в MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    image_tensor = transform(image_inverted).unsqueeze(0).to(device)  # Добавить batch dimension
    
    # Предсказание
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
        confidence = F.softmax(output, dim=1).max().item()
    
    # Показать изображение и результат
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    processed_image = transforms.Resize((28, 28))(transforms.Grayscale()(image))
    plt.imshow(processed_image, cmap='gray')
    plt.title(f'Предсказание: {prediction}\nУверенность: {confidence:.2%}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return prediction, confidence

# Функция для загрузки сохраненной модели
def load_model(model_path, device):
    model = DigitRecognitionNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def run_train_and_test(device):
    # Параметры
    batch_size = 64
    epochs = 5
    lr = 0.01
    
    
    # Преобразования данных
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Загрузка данных MNIST
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Создание модели и оптимизатора
    model = DigitRecognitionNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Обучение и тестирование
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    # Сохранение модели
    torch.save(model.state_dict(), "digit_recognition_model.pth")
    print("Модель сохранена как digit_recognition_model.pth")
    
# Основная функция
def main():
    # Проверка доступности GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        try:
            User_Input = int(input("Выберите режим:\n1. Обучение и тестирование модели\n2. Тестирование собственного изображения\n0. Выход\nВведите 1, 2 или 0: "))
        except ValueError:
            print("Введите корректное число!")
            continue
            
        if User_Input == 1:
            run_train_and_test(device)
        elif User_Input == 2:
            try:
                image_path = input("Введите путь к изображению: ")
                model = load_model("digit_recognition_model.pth", device)
                predict_custom_image(model, device, image_path)
            except FileNotFoundError:
                print("Файл модели или изображения не найден!")
            except Exception as e:
                print(f"Ошибка: {e}")
        elif User_Input == 0:
            print("Выход из программы")
            break
        else:
            print("Введите 1, 2 или 0!")
            
if __name__ == '__main__':
    main()