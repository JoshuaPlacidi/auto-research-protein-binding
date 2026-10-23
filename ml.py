import logging
import os

import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger("training")

EXPERIMENT_DIR = "experiments/1/"

MAX_PEPTIDE_LENGTH = 24
MAX_HLA_LENGTH = 366
MAX_TCR_LENGTH = 247

# Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path, low_memory=False)
        self.tokenizer = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20}

    def tokenize(self, text: str):
        return [self.tokenizer[char] for char in text]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data.iloc[idx]
        
        # inputs
        peptide = torch.tensor(self.tokenize(x["Peptide"]))
        hla = torch.tensor(self.tokenize(x["HLA_sequence"]))
        tcr = torch.tensor(self.tokenize(x["TCR_full"]))

        # pad inputs
        peptide = torch.nn.functional.pad(peptide, (0, MAX_PEPTIDE_LENGTH - len(peptide)), mode="constant", value=0)
        hla = torch.nn.functional.pad(hla, (0, MAX_HLA_LENGTH - len(hla)), mode="constant", value=0)
        tcr = torch.nn.functional.pad(tcr, (0, MAX_TCR_LENGTH - len(tcr)), mode="constant", value=0)

        # outputs
        binding_flag = torch.tensor(x["binding_flag"])
        
        return peptide, hla, tcr, binding_flag

# Model
class Model(torch.nn.Module):
    def __init__(self, embedding_dim: int = 768):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(21, embedding_dim)
        self.fc1 = torch.nn.Linear(embedding_dim, 1)

    def forward(self, peptide, hla, tcr):
        peptide = self.embedding(peptide)
        hla = self.embedding(hla)
        tcr = self.embedding(tcr)

        x = torch.cat([peptide, hla, tcr], dim=1)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        return x

# Training
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs):

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            peptide, hla, tcr, binding_flag = batch
            pred = model(peptide.to(device), hla.to(device), tcr.to(device)).squeeze(-1)
            loss = criterion(pred, binding_flag.to(device).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss, auroc, auprc = evaluate_model(model, val_loader, device, criterion)
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

    return model

def evaluate_model(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            peptide, hla, tcr, binding_flag = batch
            pred = model(peptide.to(device), hla.to(device), tcr.to(device)).squeeze(-1)
            loss = criterion(pred, binding_flag.to(device).float())
            val_loss += loss.item()
            preds.append(pred.flatten().cpu())
            labels.append(binding_flag.flatten().cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    return val_loss / len(val_loader), auroc, auprc

def configure_logger(experiment_dir):
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(experiment_dir, "training.log"), mode="w")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def main():

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    configure_logger(EXPERIMENT_DIR)

    train_dataset = Dataset(os.getenv("TRAIN_PATH"))
    val_dataset = Dataset(os.getenv("VALIDATION_PATH"))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model = train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10)

    torch.save(model.state_dict(), os.path.join(EXPERIMENT_DIR, "model.pth"))


if __name__ == "__main__":
    main()