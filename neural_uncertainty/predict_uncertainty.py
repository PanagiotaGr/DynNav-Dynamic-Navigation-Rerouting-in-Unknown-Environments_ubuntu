import torch
import numpy as np
import torch.nn as nn
import joblib

class UncertaintyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)

# ✅ Load model
model = UncertaintyNet()
model.load_state_dict(torch.load("uncertainty_net.pt"))
model.eval()

# ✅ Load SAME scaler
scaler = joblib.load("uncertainty_scaler.save")

# Example input
sample = np.array([[110, 0.06, 0.25, 0.04, 0.3]], dtype=np.float32)

# ✅ Normalize correctly
sample_norm = scaler.transform(sample)
sample_t = torch.tensor(sample_norm, dtype=torch.float32)

with torch.no_grad():
    pred = model(sample_t).item()

print(f"✅ Predicted future uncertainty: {pred:.4f}")
