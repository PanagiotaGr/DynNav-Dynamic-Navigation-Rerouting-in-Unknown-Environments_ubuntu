import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from learned_heuristic import HeuristicNet


def online_update(model_path: str = "heuristic_net_rich.pt",
                  log_path: str = "heuristic_logs.npz",
                  epochs: int = 3,
                  lr: float = 1e-4,
                  batch_size: int = 128):
    """
    Online fine-tuning του heuristic μοντέλου πάνω στα logged samples.
    """
    data = np.load(log_path)
    X = data["X"]
    y = data["y"]

    print(f"[ONLINE] Loaded {X.shape[0]} samples from {log_path}")
    input_dim = X.shape[1]

    model = HeuristicNet(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.train()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        losses = []
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"[ONLINE] Epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"[ONLINE] Updated model saved to {model_path}")


if __name__ == "__main__":
    online_update()
