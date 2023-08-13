import cv2
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision.transforms import transforms
from train import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = Path("neural_networks/weights")
data_path = Path("data/by_character")
data_paths = [data_path / f"{chr(65 + i)}_cap" for i in range(26)] + [data_path / f"{chr(97 + i)}_low" for i in range(26)] + [data_path / f"{i}_num" for i in range(10)]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# Load all the trained models
models = []
for data_path in data_paths:
    model = Trainer().to(device)
    weights = weights_path / f"weights_{data_path.stem}.pth"
    model.load_state_dict(torch.load(weights))
    model.eval()
    models.append(model)

# Initialize the webcam
cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    pil_image = Image.fromarray(frame)

    # Preprocess the frame
    preprocessed_frame = transform(pil_image).unsqueeze(0).to(device)

    # Make predictions with all the models
    predictions = []
    with torch.no_grad():
        for model in models:
            outputs = model(preprocessed_frame)
            _, predicted = torch.max(outputs.data, 1)
            predicted_letter = chr(ord('A') + predicted.item()) if predicted.item() < 26 else chr(ord('a') + predicted.item() - 26)
            predictions.append(predicted_letter)

    # Get the majority predicted letter
    predicted_letter = max(set(predictions), key=predictions.count)

    # Display the predicted letter on the frame
    cv2.putText(frame, predicted_letter, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Letter Recognition', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy the windows
cap.release()
cv2.destroyAllWindows()
