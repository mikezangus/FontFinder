import matplotlib.pyplot as plt
import os
import random
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from train import NN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = Path("data/by_character")
weights_path = Path("models/weights")
characters = [chr(65 + i) for i in range(26)] + [str(i) for i in range(10)]
character_types = ["cap", "low", "num"]

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
    
def load_test_character():
    test_character = random.choice(characters)
    if test_character.isalpha():
        test_character_type = random.choice(character_types[:2])
    else:
        test_character_type = character_types[2]
    test_character_path = data_path / f"{test_character}_{test_character_type}"
    image_files = list(test_character_path.glob("*.png"))
    test_character_file = random.choice(image_files)
    test_image = Image.open(test_character_file).convert("L")
    test_image_tensor = transform(test_image).unsqueeze(0).to(device)
    print(f"Testing {test_character} {test_character_type} via {test_character_file}")
    return test_image, test_character, test_character_type, test_image_tensor

def main():
    model = NN().to(device)
    model.eval()
    fig, axes = plt.subplots(3, 5, figsize = (5, 5))
    for row in range(3):
        for col in range(5):
            ax = axes[row, col]
            test_image, test_character, test_character_type, test_image_tensor = load_test_character()
            prediction_scores = []
            for character in characters:
                for character_type in character_types:
                    if (character_type == "num" and character.isalpha()) or (character_type != "num" and character.isdigit()):
                        continue
                    weight_path = weights_path / f"weights_{character}_{character_type}.pth"
                    other_model = NN().to(device)
                    other_model.load_state_dict(torch.load(weight_path))
                    other_model.eval()
                    with torch.no_grad():
                        other_output = other_model(test_image_tensor)
                    predicted_value = torch.argmax(other_output.data, 1).item()
                    certainty = (1 - torch.softmax(other_output.data, 1)[0][predicted_value].item())
                    prediction_scores.append((character, character_type, certainty))
            closest_predictions = sorted(prediction_scores, key = lambda x: x[2], reverse = True)[:3]
            predicted_characters = [f"{character} {character_type}" for character, character_type, _ in closest_predictions]
            certainties = [certainty for _, _, certainty in closest_predictions]
            ax.imshow(test_image, cmap = "gray")
            title_text = f"Actual: {test_character} {test_character_type}\nTop Predictions:\n{predicted_characters[0]} - Certainty: {certainties[0]:.4f}\n{predicted_characters[1]} - Certainty: {certainties[1]:.2f}\n{predicted_characters[2]} - Certainty: {certainties[2]:.2f}"
            ax.set_title(title_text)
            ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

