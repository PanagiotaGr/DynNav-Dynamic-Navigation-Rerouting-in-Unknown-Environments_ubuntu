import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("uncertainty_dataset.csv")

X = df[["inliers", "drift", "lin_vel", "ang_vel", "entropy"]].values
y = df[["target_uncertainty"]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Neural network
class UncertaintyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()   # ✅ εγγυάται θετική αβεβαιότητα
        )

    def forward(self, x):
        return self.net(x)

model = UncertaintyNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.6f}")

# Evaluation
with torch.no_grad():
    test_loss = criterion(model(X_test), y_test).item()

print(f"✅ Test MSE: {test_loss:.6f}")

# Save model
torch.save(model.state_dict(), "uncertainty_net.pt")
print("✅ Model saved to uncertainty_net.pt")

import joblib
joblib.dump(scaler, "uncertainty_scaler.save")
print("✅ Scaler saved to uncertainty_scaler.save")
