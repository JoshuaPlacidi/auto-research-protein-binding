import logging
import os

import torch
from ml import Model, Dataset
from sklearn.metrics import average_precision_score, roc_auc_score

from tqdm import tqdm

logger = logging.getLogger("evaluation")

def configure_logger(experiment_dir):
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(experiment_dir, "evaluation.log"), mode="w")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    preds, labels = [], []
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            peptide, hla, tcr, binding_flag = batch
            pred = model(peptide.to(device), hla.to(device), tcr.to(device)).squeeze(-1)
            loss = criterion(pred, binding_flag.to(device).float())
            test_loss += loss.item()
            preds.append(pred.flatten().cpu())
            labels.append(binding_flag.flatten().cpu())
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    return test_loss / len(test_loader), auroc, auprc

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    args = parser.parse_args()

    configure_logger(args.experiment_dir)

    model = Model()
    model.load_state_dict(torch.load(os.path.join(args.experiment_dir, "model.pth")))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset = Dataset(os.getenv("TEST_PATH"))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    test_loss, auroc, auprc = evaluate_model(model, test_loader, device)
    logger.info(f"Test Loss: {test_loss}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
