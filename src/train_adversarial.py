import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.data_loader import DeepfakeSequenceDataset
from src.model import CNNFeatureExtractor, CNN_LSTM_Attention
from src.adversarial import fgsm_attack


def train_epoch_adv(feat_extractor, model, dataloader, criterion, optimizer, device, epsilon):
    feat_extractor.train()
    model.train()

    losses = []
    ys = []
    ypred = []

    for seqs, labels in dataloader:
        seqs = seqs.to(device)
        labels = labels.to(device)

        # ---- Clean forward pass ----
        B, S, C, H, W = seqs.shape
        seqs_flat = seqs.view(B*S, C, H, W)
        feats = feat_extractor(seqs_flat)
        feats = feats.view(B, S, -1)
        logits, _ = model(feats)

        # ---- Generate adversarial examples ----
        seqs_adv = fgsm_attack(feat_extractor, model, seqs, labels, epsilon)

        B, S, C, H, W = seqs_adv.shape
        seqs_adv_flat = seqs_adv.view(B*S, C, H, W)
        feats_adv = feat_extractor(seqs_adv_flat)
        feats_adv = feats_adv.view(B, S, -1)
        logits_adv, _ = model(feats_adv)

        # ---- Combine clean + adversarial loss ----
        loss_clean = criterion(logits, labels)
        loss_adv = criterion(logits_adv, labels)
        loss = (loss_clean + loss_adv) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        ys.extend(labels.cpu().numpy())
        ypred.extend(preds.tolist())

    acc = accuracy_score(ys, ypred)
    return sum(losses)/len(losses), acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu"

    dataset = DeepfakeSequenceDataset(args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    feat_extractor = CNNFeatureExtractor().to(device)
    model = CNN_LSTM_Attention(feat_dim=feat_extractor.out_dim).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    feat_extractor.load_state_dict(checkpoint["feat_state"])
    model.load_state_dict(checkpoint["model_state"])

    optimizer = optim.Adam(list(model.parameters()) + list(feat_extractor.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loss, acc = train_epoch_adv(feat_extractor, model, dataloader,
                                    criterion, optimizer, device, args.epsilon)
        print(f"Epoch {epoch+1}/{args.epochs}  Loss: {loss:.4f}  Acc: {acc:.4f}")

    os.makedirs("experiments/checkpoints", exist_ok=True)
    torch.save({
        "feat_state": feat_extractor.state_dict(),
        "model_state": model.state_dict()
    }, "experiments/checkpoints/best_adv.pth")

    print("\nAdversarial fine-tuning completed.")
