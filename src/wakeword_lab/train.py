"""Train the wake-word classifier and export it as an openWakeWord ONNX model.

The model is deliberately tiny (about 100k parameters): a two-layer MLP over
the flattened 16x96 embedding window. openWakeWord's pretrained embedding
model already does the heavy lifting, so this head trains in seconds to a
few minutes and runs comfortably on a Raspberry Pi.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import EMBEDDING_DIM, N_FRAMES
from .config import TrainingSettings


def build_model(hidden: int):
    from torch import nn

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(N_FRAMES * EMBEDDING_DIM, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
        nn.Sigmoid(),
    )


def window_metrics(pred: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, recall and false-positive rate on individual windows."""
    pred = np.asarray(pred).reshape(-1) >= threshold
    y = np.asarray(y).reshape(-1) > 0.5
    pos, neg = int(y.sum()), int((~y).sum())
    return {
        "accuracy": float((pred == y).mean()) if y.size else 0.0,
        "recall": float((pred & y).sum() / pos) if pos else 0.0,
        "false_positive_rate": float((pred & ~y).sum() / neg) if neg else 0.0,
    }


def train(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    settings: TrainingSettings,
    out_path: Path,
    output_name: str,
    log=print,
) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(settings.hidden_dim).to(device)
    xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device).unsqueeze(1)
    has_val = len(y_val) > 0
    if has_val:
        xv = torch.from_numpy(X_val).to(device)
        yv = torch.from_numpy(y_val).to(device).unsqueeze(1)

    # A wake word that fires by mistake is worse than one that needs a repeat,
    # so false positives cost more than misses.
    neg_w = torch.tensor([settings.negative_weight], device=device)
    pos_w = torch.tensor([1.0], device=device)
    loss_fn = nn.BCELoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    def weighted_loss(pred, target):
        return (loss_fn(pred, target) * torch.where(target > 0.5, pos_w, neg_w)).mean()

    n = xt.shape[0]
    batch = min(settings.batch_size, n)
    history: list[dict] = []
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(1, settings.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            loss = weighted_loss(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.shape[0]

        model.eval()
        with torch.no_grad():
            row = {"epoch": epoch, "train_loss": total / n}
            if has_val:
                pv = model(xv)
                row["val_loss"] = float(weighted_loss(pv, yv))
                row.update({f"val_{k}": v for k, v in window_metrics(pv.cpu().numpy(), y_val).items()})
        history.append(row)

        score = row.get("val_loss", row["train_loss"])
        if score < best_loss:
            best_loss, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == settings.epochs:
            extra = f"  val_loss={row['val_loss']:.4f}  val_recall={row['val_recall']:.3f}" if has_val else ""
            log(f"    epoch {epoch:02d}/{settings.epochs}  loss={row['train_loss']:.4f}{extra}")
        if has_val and stale >= settings.patience:
            log(f"    early stop at epoch {epoch} (best val_loss={best_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.cpu(),
        torch.zeros(1, N_FRAMES, EMBEDDING_DIM),
        str(out_path),
        input_names=["input"],
        output_names=[output_name],
        dynamic_axes={"input": {0: "batch"}, output_name: {0: "batch"}},
        opset_version=13,
    )

    report = {
        "device": device.type,
        "train_windows": {"positive": int((y == 1).sum()), "negative": int((y == 0).sum())},
        "holdout_windows": {"positive": int((y_val == 1).sum()), "negative": int((y_val == 0).sum())},
        "best_epoch": min(history, key=lambda r: r.get("val_loss", r["train_loss"]))["epoch"],
        "history": history,
    }
    out_path.with_suffix(".train.json").write_text(json.dumps(report, indent=2))
    log(f"  saved {out_path}")
    return report
