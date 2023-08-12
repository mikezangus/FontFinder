import matplotlib.pyplot as plt
import os
import random
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from characters_nn import NeuralNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = Path("data/by_character")
weights_path = Path("neural_networks/weights")

input_letter = "Z"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 1),
    transforms.Resize((50, 50)),
    transforms.CenterCrop(50),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485], std = [0.229])
])

class CustomDataset(Dataset):
    def __init__(self, data, transform = None):
        self.transform = transform
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img

def test(image_path, model):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted.item() == 0:
            return f"{input_letter}"
        else:
            return f"Not {input_letter}"

if __name__ == "__main__":
    weights = weights_path / f"weights_{input_letter}_cap.pth"
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    grid_size = (6, 9)
    spacing = 0.69
    fig_width = grid_size[1] * (1 + spacing)
    fig_height = grid_size[0] * (1 + spacing)
    fig, ax = plt.subplots(grid_size[0], grid_size[1], figsize = (fig_width, fig_height))
    letters = [chr(65 + i) for i in range(26)]
    random_image_paths = []
    while len(random_image_paths) < grid_size[0] * grid_size[1]:
        test_letter = random.choice(letters)
        image_files = os.listdir(data_path / f"{test_letter}_cap")
        random_image_path = random.choice(image_files)
        random_image_paths.append((test_letter, random_image_path))
    for test, (test_letter, image_path) in enumerate(random_image_paths):
        image = Image.open(data_path / f"{test_letter}_cap" / image_path).convert("L")
        image = transform(image).unsqueeze(0).to(device)
        predicted_value = torch.argmax(model(image)).item()
        predicted_label = f"{input_letter}" if predicted_value == 0 else f"not {input_letter}"
        label_color = "blue" if predicted_value == 0 else "gray"
        x_index = test % grid_size[1]
        y_index = test // grid_size[1]
        ax[y_index, x_index].imshow(transforms.ToPILImage()(image.squeeze()), cmap = "gray")
        ax[y_index, x_index].set_xticks([])
        ax[y_index, x_index].set_yticks([])
        ax[y_index, x_index].set_title(f"Predicted: {predicted_label}\nActual: {test_letter}", color = label_color)
    plt.tight_layout()
    plt.show()