import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = Path("neural_networks/weights")
data_path = Path("data/by_character")
data_paths = [data_path / f"{chr(65 + i)}_cap" for i in range(26)] + [data_path / f"{chr(97 + i)}_low" for i in range(26)] + [data_path / f"{i}_num" for i in range(10)]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((50, 50)),
    transforms.RandomCrop(50),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485], std = [0.229])
])

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform = None):
        self.transform = transform
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Flatten(),
            nn.Linear(in_features = 8 * 25 * 25, out_features = 2)
        ).to(device)
    def forward(self, x):
        return self.layers(x)

class Trainer():
    def __init__(self, data_paths, transform, device):
        self.data_paths = data_paths
        self.transform = transform
        self.device = device
    def train(self):
        criterion = nn.CrossEntropyLoss()
        for i, data_path in enumerate(self.data_paths):
            data = []
            targets = []
            for j, path in enumerate(self.data_paths):
                for filename in os.listdir(path):
                    if filename.endswith(".png"):
                        img_path = os.path.join(path, filename)
                        data.append(img_path)
                        targets.append(int(i != j))

            train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2, random_state=42)
            train_dataset = CustomDataset(train_data, train_targets, self.transform)
            test_dataset = CustomDataset(test_data, test_targets, self.transform)
            train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
            test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

            model = NeuralNet().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr  = 0.001)

            num_epochs = 1
            for epoch in range(num_epochs):
                train_loss = 0
                train_correct = 0
                total_train = 0
                start_time = time.time()
                print(f"Training {data_path.stem}")
                model.train()
                for images, labels in train_dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_correct += (predicted == labels).sum().item()
                    total_train += labels.size(0)
                end_time = time.time()
                epoch_time = end_time - start_time
                train_accuracy = 100 * train_correct / total_train
                train_loss /= len(train_dataloader)
                test_loss = 0
                test_correct = 0
                total_test = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in test_dataloader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        test_correct += (predicted == labels).sum().item()
                        total_test += labels.size(0)
                test_accuracy = 100 * test_correct / total_test
                test_loss /= len(test_dataloader)
                print(f"Character: {data_path.stem} | Epoch: {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f} | Epoch Time: {epoch_time / 60 :.2f} mins")
            weights = weights_path / f"weights_{data_path.stem}.pth"
            torch.save(model.state_dict(), weights)
            print(f"Saved weights to {weights}")

if __name__ == "__main__":
    trainer = Trainer(data_paths, transform, device)
    trainer.train()