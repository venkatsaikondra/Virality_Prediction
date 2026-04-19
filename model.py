import torch
import torch.nn as nn
from transformers import AutoModel

class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")

        self.numeric_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, numeric):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = text_out.last_hidden_state[:, 0, :]

        num_out = self.numeric_net(numeric)

        combined = torch.cat((cls, num_out), dim=1)

        return self.classifier(combined)