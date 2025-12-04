import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from learned_heuristic import HeuristicNet


class NPZPlannerDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        print(f"[INFO] Loaded npz from '{npz_path}'")
        print(f"[INFO] Keys: {data.files}")

        X = data["X"]
        y = data["y"]

        print(f"[INFO] X shape: {X.shape}")
        print(f"[INFO] y shape: {y.shape}")

        self.features = X.astype("float32")
        y = y.reshape(-1, 1)
        self.targets = y.astype("float32")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.targets[idx]),
        )


def train_heuristic_from_npz(npz_path: str,
                             model_out: str = "heuristic_net_rich.pt",
                             batch_size: int = 128,
                             lr: float = 1e-3,
                             epochs: int = 15):

    dataset = NPZPlannerDataset(npz_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = dataset.features.shape[1]
    print(f"[INFO] Input dim = {input_dim}")

    model = HeuristicNet(input_dim=input_dim)
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
    print(f"[INFO] Saved trained heuristic to: {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="planner_dataset_rich.npz")
    parser.add_argument("--out", type=str, default="heuristic_net_rich.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    train_heuristic_from_npz(
        npz_path=args.npz,
        model_out=args.out,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
    )
