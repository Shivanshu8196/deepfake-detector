import torch
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.data_loader import DeepfakeSequenceDataset
from src.model import CNNFeatureExtractor, CNN_LSTM_Attention
from src.adversarial import fgsm_attack, pgd_attack


def evaluate(feat_extractor, model, dataloader, device, attack=None, epsilon=0.03):
    feat_extractor.eval()
    model.eval()

    all_preds = []
    all_labels = []

    for seqs, labels in dataloader:
        seqs = seqs.to(device)
        labels = labels.to(device)

        if attack == "fgsm":
            seqs = fgsm_attack(feat_extractor, model, seqs, labels, epsilon)
        elif attack == "pgd":
            seqs = pgd_attack(feat_extractor, model, seqs, labels, epsilon)

        B, S, C, H, W = seqs.shape
        seqs_flat = seqs.view(B*S, C, H, W)

        feats = feat_extractor(seqs_flat)
        feats = feats.view(B, S, -1)

        logits, _ = model(feats)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--attack", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu"

    dataset = DeepfakeSequenceDataset(args.data_root)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    feat_extractor = CNNFeatureExtractor().to(device)
    model = CNN_LSTM_Attention(feat_dim=feat_extractor.out_dim).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    feat_extractor.load_state_dict(checkpoint["feat_state"])
    model.load_state_dict(checkpoint["model_state"])

    acc = evaluate(feat_extractor, model, dataloader, device,
                   attack=args.attack, epsilon=args.epsilon)

    print(f"\nAccuracy ({args.attack if args.attack else 'normal'}): {acc:.4f}")
