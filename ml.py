import torch
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

EXPERIMENT_DIR = "experiments/1/"

# Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.tokenizer = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19}

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

        # outputs
        binding_flag = torch.tensor(x["binding_flag"])
        
        return peptide, hla, tcr, binding_flag

# Model
class Model(torch.nn.Module):
    def __init__(self, embedding_dim: int = 10):
        super(Model, self).__init__()
        self.embedding = torch.nn.Embedding(20, 10)
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, peptide, hla, tcr):
        peptide = self.embedding(peptide)
        hla = self.embedding(hla)
        tcr = self.embedding(tcr)
        print(peptide.shape, hla.shape, tcr.shape)
        x = torch.cat([peptide, hla, tcr], dim=1)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        input()
        return torch.tensor(0).to(peptide.device)

# Training
def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs):

    model.train()

    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            peptide, hla, tcr, binding_flag = batch
            pred = model(peptide.to(device), hla.to(device), tcr.to(device))
            loss = criterion(pred, binding_flag)
            loss.backward()
            optimizer.step()
            
            val_loss = evaluate_model(model, val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Val Loss: {val_loss}")

    return model

def main():

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    train_dataset = Dataset(os.getenv("TRAIN_PATH"))


    print(train_dataset[0])
    val_dataset = Dataset(os.getenv("VALIDATION_PATH"))

    train_loader = torch.utils.data.DataLoader(train_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model = train_model(model, train_loader, val_loader, optimizer, criterion, device, 2)

    torch.save(model.state_dict(), os.path.join(EXPERIMENT_DIR, "model.pth"))


if __name__ == "__main__":
    main()