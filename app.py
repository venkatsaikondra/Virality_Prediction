import torch
import numpy as np
import joblib
from transformers import AutoTokenizer
from model import MultiModalModel

device = torch.device("cpu")  # or "cuda" if GPU available

# Load model
model = MultiModalModel().to(device)
model.load_state_dict(torch.load("model/model.pt", map_location=device))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Load threshold
with open("model/tokenizer/threshold.txt", "r") as f:
    threshold = float(f.read())


def predict_viral(title, comments, age_hours):

    encoding = tokenizer(
        title,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    numeric = np.array([[age_hours, np.log1p(comments)]])
    numeric = scaler.transform(numeric)
    numeric = torch.tensor(numeric, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
            numeric
        )

    prob = torch.sigmoid(output).item()
    label = int(prob > threshold)

    return prob, label


# 🔥 Test
if __name__ == "__main__":
    prob, label = predict_viral(
        "Amazing data visualization about global warming",
        50,
        2
    )

    print("Probability:", prob)
    print("Predicted:", label)