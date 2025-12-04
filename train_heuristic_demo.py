import numpy as np
import pandas as pd
import torch
import argparse
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- 1. Δημιουργία dummy CSV (εσύ εδώ μετά θα βάλεις το planner_dataset σου) ----------

def create_dummy_planner_csv(path: str, num_samples: int = 500, num_features: int = 16):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(num_samples, num_features))
    true_w = rng.normal(size=(num_features, 1))
    y = X @ true_w + rng.normal(scale=0.1, size=(num_samples, 1))

    columns = [f"feature_{i}" for i in range(num_features)] + ["target_cost"]
    data = np.hstack([X, y])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(path, index=False)
    print(f"Saved dummy dataset to: {path}")


# ---------- 2. Dataset wrapper ----------

class PlannerDataset(Dataset):
    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)
        self.feature_cols = [c for c in self.df.columns if c.startswith("feature_")]
        self.features = self.df[self.feature_cols].values.astype("float32")
        self.targets = self.df["target_cost"].values.astype("float32").reshape(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ---------- 3. Heuristic network (MLP) ----------

class HeuristicNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# ---------- 4. Training loop ----------

def train_heuristic(csv_path: str,
                    model_out: str = "heuristic_net.pt",
                    batch_size: int = 32,
                    lr: float = 1e-3,
                    epochs: int = 10):

    dataset = PlannerDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = HeuristicNet(input_dim=len(dataset.feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(dataset)
        print(f"[Epoch {epoch+1}/{epochs}] Loss = {epoch_loss:.4f}")

    torch.save(model.state_dict(), model_out)
    print(f"Saved trained heuristic to: {model_out}")


# ---------- 5. Main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="dummy_planner_dataset.csv",
                        help="Path to planner dataset CSV")
    parser.add_argument("--out", type=str, default="heuristic_net_demo.pt",
                        help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_dummy", action="store_true",
                        help="If set, create a dummy CSV instead of using an existing one.")
    args = parser.parse_args()

    if args.use_dummy:
        # Δημιουργεί ΠΑΝΤΑ dummy data (όπως πριν)
        create_dummy_planner_csv(args.csv)
    else:
        # Δεν ακουμπάει ΚΑΘΟΛΟΥ το CSV· απλώς ελέγχει ότι υπάρχει
        if not os.path.exists(args.csv):
            raise FileNotFoundError(
                f"CSV file '{args.csv}' not found. "
                "Either run with --use_dummy or give an existing CSV."
            )

    train_heuristic(args.csv,
                    model_out=args.out,
                    lr=args.lr,
                    epochs=args.epochs)
