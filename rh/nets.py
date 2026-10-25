"""Small networks and a generic training loop (GPU if available)."""
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, n_in, n_out=1, width=64, depth=3, act=nn.SiLU):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEV)


def train(model, loss_fn, epochs=2000, lr=3e-3, wd=0.0, log_every=0):
    """Full-batch Adam with cosine decay. loss_fn() -> scalar tensor."""
    model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item())
        if log_every and ep % log_every == 0:
            print(f"  ep {ep:5d} loss {hist[-1]:.4e}")
    return hist


def fit_mlp(X, y, epochs=3000, lr=3e-3, **kw):
    X, y = tensor(X), tensor(y).reshape(len(X), -1)
    model = MLP(X.shape[1], y.shape[1], **kw).to(DEV)
    hist = train(model, lambda: torch.mean((model(X) - y) ** 2), epochs, lr)
    return model, hist


def predict(model, X):
    with torch.no_grad():
        return model(tensor(X)).cpu().numpy()
