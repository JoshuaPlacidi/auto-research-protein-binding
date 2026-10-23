"""Sequence-only baseline. Train only on eligible complete alpha+beta rows.

Dataset.data preserves CSV order for the fixed evaluator. Training/validation
loaders select eligible_indices explicitly; no sequences are cropped.
"""
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv

from eval import eligible_rows, metric_summary

load_dotenv()
EXPERIMENT_DIR = os.getenv("EXPERIMENT_DIR", "experiments/1/")


def log_event(event, **fields):
    path = Path(os.getenv("RESEARCH_LOG_PATH", "experiments/evaluation.log"))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(),
              "experiment_id": os.getenv("RESEARCH_EXPERIMENT_ID", "manual"), **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


def save_checkpoint(model, name):
    """Atomic replacement avoids corrupting an existing checkpoint on interruption."""
    directory = Path(EXPERIMENT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, path)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        if not data_path:
            raise ValueError("Dataset CSV path is required")
        self.data = pd.read_csv(data_path, low_memory=False)
        cohort, self.eligible_indices = eligible_rows(self.data)
        if cohort.empty:
            raise ValueError("No eligible full-chain rows")
        self.eligible_set = set(self.eligible_indices.tolist())
        self.tokenizer = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        # Dataset-local lengths are safe because Model masks padding. Every
        # sequence is preserved, including long sequences in a new test CSV.
        self.max_lengths = {c: int(cohort[c].str.len().max())
                            for c in ("Peptide", "HLA_sequence", "TCR_full")}

    def __len__(self):
        return len(self.data)

    def tokenize_and_pad(self, sequence, column):
        try:
            values = [self.tokenizer[aa] for aa in sequence]
        except KeyError as exc:
            raise ValueError(f"Unsupported amino-acid symbol in {column}; do not silently replace it") from exc
        padding = self.max_lengths[column] - len(values)
        if padding < 0:
            raise ValueError("Input exceeds padding length; cropping is forbidden")
        return torch.tensor(values + [0] * padding, dtype=torch.long)

    def __getitem__(self, index):
        index = int(index)
        if index not in self.eligible_set:
            raise ValueError("Single-chain/ineligible row requested; use the full-chain Subset")
        row = self.data.iloc[index]
        return (self.tokenize_and_pad(row.Peptide, "Peptide"),
                self.tokenize_and_pad(row.HLA_sequence, "HLA_sequence"),
                self.tokenize_and_pad(row.TCR_full, "TCR_full"),
                torch.tensor(float(row.binding_flag), dtype=torch.float32))


class Model(torch.nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.embedding = torch.nn.Embedding(21, embedding_dim, padding_idx=0)
        self.fc1 = torch.nn.Linear(embedding_dim, 1)

    def forward(self, peptide, hla, tcr):
        tokens = torch.cat([peptide, hla, tcr], dim=1)
        mask = tokens.ne(0).unsqueeze(-1)
        embeddings = self.embedding(tokens)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return torch.sigmoid(self.fc1(pooled))


def evaluate_model(model, val_loader, device, criterion):
    model.eval()
    predictions, labels = [], []
    loss_sum, count = 0.0, 0
    with torch.no_grad():
        for peptide, hla, tcr, target in val_loader:
            target = target.to(device)
            score = model(peptide.to(device), hla.to(device), tcr.to(device)).squeeze(-1)
            loss_sum += criterion(score, target).item() * len(target)
            count += len(target)
            predictions.extend(score.cpu().tolist())
            labels.extend(target.cpu().tolist())
    metrics = metric_summary(labels, predictions)
    return loss_sum / count, metrics["auroc"], metrics["auprc"]


def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    best_loss = float("inf")
    try:
        for epoch in range(epochs):
            model.train()
            loss_sum, count = 0.0, 0
            for peptide, hla, tcr, target in train_loader:
                optimizer.zero_grad()
                target = target.to(device)
                score = model(peptide.to(device), hla.to(device), tcr.to(device)).squeeze(-1)
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(target)
                count += len(target)
            val_loss, auroc, auprc = evaluate_model(model, val_loader, device, criterion)
            save_checkpoint(model, "last_model.pth")
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(model, "model.pth")
            log_event("training_epoch", epoch=epoch+1, train_loss=loss_sum/count,
                      validation_loss=val_loss, auroc=auroc, auprc=auprc,
                      best_validation_loss=best_loss)
            print(f"Epoch {epoch+1}/{epochs}: validation loss={val_loss:.6f}, AUROC={auroc}", flush=True)
    finally:
        # Graceful SIGINT/timeout preserves the latest available weights even
        # if an epoch is incomplete. It is not a selected/validated checkpoint.
        save_checkpoint(model, "last_model.pth")
    return model


def main():
    import numpy as np
    seed = int(os.getenv("RESEARCH_SEED", "31"))
    epochs = int(os.getenv("RESEARCH_EPOCHS", "10"))
    if epochs < 1:
        raise ValueError("RESEARCH_EPOCHS must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)
    train_dataset = Dataset(os.getenv("TRAIN_PATH"))
    val_dataset = Dataset(os.getenv("VALIDATION_PATH"))
    train_subset = torch.utils.data.Subset(train_dataset, train_dataset.eligible_indices.tolist())
    val_subset = torch.utils.data.Subset(val_dataset, val_dataset.eligible_indices.tolist())
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=128, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    save_checkpoint(model, "initial_model.pth")
    log_event("training_configuration", seed=seed, epochs=epochs, learning_rate=1e-4,
              batch_size=128, loss="MSE", train_rows=len(train_subset),
              validation_rows=len(val_subset), checkpoint_directory=str(EXPERIMENT_DIR))
    train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs)


if __name__ == "__main__":
    main()
