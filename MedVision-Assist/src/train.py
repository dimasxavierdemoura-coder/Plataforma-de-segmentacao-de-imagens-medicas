import csv
import os
import sys
import json
import random
import argparse
import platform
import time
from datetime import datetime
from torch.utils.data import DataLoader
from torch import nn, optim

# Adicionar o diretório raiz ao path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.unet import UNet
from src.data.preprocessing import MedicalImageDataset, build_dataset_from_root, split_dataset, validate_dataset
from src.utils import dice_coefficient, iou_score, sensitivity_score, specificity_score, f1_score


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_score(output, target, smooth=1e-6):
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    intersection = (output * target).sum(dim=(1, 2, 3))
    union = output.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + smooth) / (union + smooth)).mean().item()


def iou_score(output, target, smooth=1e-6):
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    intersection = (output * target).sum(dim=(1, 2, 3))
    union = (output + target - output * target).sum(dim=(1, 2, 3))
    return ((intersection + smooth) / (union + smooth)).mean().item()


def sensitivity_score(output, target, smooth=1e-6):
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    tp = (output * target).sum(dim=(1, 2, 3))
    fn = ((1 - output) * target).sum(dim=(1, 2, 3))
    return ((tp + smooth) / (tp + fn + smooth)).mean().item()


def specificity_score(output, target, smooth=1e-6):
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    tn = ((1 - output) * (1 - target)).sum(dim=(1, 2, 3))
    fp = (output * (1 - target)).sum(dim=(1, 2, 3))
    return ((tn + smooth) / (tn + fp + smooth)).mean().item()


def f1_score(output, target, smooth=1e-6):
    dice = dice_score(output, target, smooth)
    return dice


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

        self.best_score = score
        self.counter = 0
        return False


def save_checkpoint(state, checkpoint_dir, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, path)
    return path


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)


def save_experiment_metadata(metadata, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    metadata_path = os.path.join(checkpoint_dir, 'experiment_metadata.json')
    save_json(metadata_path, metadata)
    return metadata_path


def save_training_config(config, checkpoint_dir, config_name='training_config.json'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    config_path = os.path.join(checkpoint_dir, config_name)
    save_json(config_path, config)
    return config_path


def evaluate_loader(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    metrics = {'dice': 0.0, 'iou': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'f1': 0.0}
    steps = 0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(next(model.parameters()).device)
            masks = batch['mask'].to(next(model.parameters()).device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            metrics['dice'] += dice_score(outputs, masks)
            metrics['iou'] += iou_score(outputs, masks)
            metrics['sensitivity'] += sensitivity_score(outputs, masks)
            metrics['specificity'] += specificity_score(outputs, masks)
            metrics['f1'] += f1_score(outputs, masks)
            steps += 1
    if steps == 0:
        return None
    return {
        'loss': total_loss / steps,
        'dice': metrics['dice'] / steps,
        'iou': metrics['iou'] / steps,
        'sensitivity': metrics['sensitivity'] / steps,
        'specificity': metrics['specificity'] / steps,
        'f1': metrics['f1'] / steps,
    }


def train_segmentation(
    train_images,
    train_masks,
    val_images=None,
    val_masks=None,
    test_images=None,
    test_masks=None,
    epochs=10,
    batch_size=4,
    lr=1e-4,
    checkpoint_dir="models/checkpoints",
    checkpoint_freq=1,
    patience=5,
    log_csv="training_log.csv",
    seed=42,
    experiment_name="medvision_experiment",
):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    validate_dataset(train_images, train_masks, max_samples=20)
    if val_images and val_masks:
        validate_dataset(val_images, val_masks, max_samples=20)
    if test_images and test_masks:
        validate_dataset(test_images, test_masks, max_samples=20)

    train_dataset = MedicalImageDataset(train_images, train_masks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if val_images and val_masks:
        val_dataset = MedicalImageDataset(val_images, val_masks)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        val_loader = None

    if test_images and test_masks:
        test_dataset = MedicalImageDataset(test_images, test_masks)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        test_loader = None

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_unet_medvision.pth")
    log_path = os.path.join(checkpoint_dir, log_csv)
    early_stopping = EarlyStopping(patience=patience) if val_loader else None
    best_val_dice = -1.0

    experiment_metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_images and val_masks else 0,
        "test_samples": len(test_dataset) if test_images and test_masks else 0,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "checkpoint_frequency": checkpoint_freq,
        "patience": patience,
    }
    save_experiment_metadata(experiment_metadata, checkpoint_dir)
    save_training_config(experiment_metadata, checkpoint_dir)

    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_dice",
                "val_iou",
                "val_sensitivity",
                "val_specificity",
                "val_f1",
                "timestamp",
            ])

    print(f"Dataset criado com {len(train_dataset)} amostras")
    print(f"DataLoader criado com batch_size={batch_size}")

    for epoch in range(1, epochs + 1):
        print(f"Iniciando epoch {epoch}/{epochs}")
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"]

            if masks is None or torch.all(masks == 0):
                continue

            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

            if batch_count % 10 == 0:
                print(f"Processado batch {batch_count}, loss atual: {loss.item():.4f}")

        train_loss = epoch_loss / max(batch_count, 1)
        print(f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f}")

        val_metrics = None
        if val_loader:
            val_metrics = evaluate_loader(model, val_loader, criterion)
            if val_metrics:
                print(f"Validation loss: {val_metrics['loss']:.4f}")
                print(f"Validation Dice: {val_metrics['dice']:.4f}")
                print(f"Validation IoU: {val_metrics['iou']:.4f}")
                print(f"Validation Sensitivity: {val_metrics['sensitivity']:.4f}")
                print(f"Validation Specificity: {val_metrics['specificity']:.4f}")
                print(f"Validation F1: {val_metrics['f1']:.4f}")

                if val_metrics['dice'] > best_val_dice:
                    best_val_dice = val_metrics['dice']
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Novo melhor modelo salvo em: {best_model_path}")

                if early_stopping and early_stopping.step(val_metrics['dice']):
                    print(f"Early stopping ativado após {epoch} epochs")
                    break

        if epoch % checkpoint_freq == 0:
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }
            checkpoint_path = save_checkpoint(checkpoint_state, checkpoint_dir, epoch)
            print(f"Checkpoint salvo em: {checkpoint_path}")

        with open(log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_metrics['loss'] if val_metrics is not None else "",
                val_metrics['dice'] if val_metrics is not None else "",
                val_metrics['iou'] if val_metrics is not None else "",
                val_metrics['sensitivity'] if val_metrics is not None else "",
                val_metrics['specificity'] if val_metrics is not None else "",
                val_metrics['f1'] if val_metrics is not None else "",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ])

    final_path = os.path.join("models", "unet_medvision.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Modelo final salvo em: {final_path}")
    if os.path.exists(best_model_path):
        print(f"Melhor modelo de validação disponível em: {best_model_path}")

    if test_loader:
        test_metrics = evaluate_loader(model, test_loader, criterion)
        if test_metrics:
            print("Avaliação de teste concluída:")
            print(f"Test loss: {test_metrics['loss']:.4f}")
            print(f"Test Dice: {test_metrics['dice']:.4f}")
            print(f"Test IoU: {test_metrics['iou']:.4f}")
            print(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
            print(f"Test Specificity: {test_metrics['specificity']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}")
            test_metrics_path = os.path.join(checkpoint_dir, 'test_metrics.json')
            save_json(test_metrics_path, test_metrics)
            print(f"Métricas de teste salvas em: {test_metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Treina a U-Net em um dataset de segmentação médica.")
    parser.add_argument("--train-root", type=str, default="data/train", help="Diretório raiz dos dados de treino")
    parser.add_argument("--val-root", type=str, default=None, help="Diretório raiz dos dados de validação")
    parser.add_argument("--test-root", type=str, default=None, help="Diretório raiz dos dados de teste")
    parser.add_argument("--train-manifest", type=str, default=None, help="Arquivo CSV manifest no diretório de treino")
    parser.add_argument("--val-manifest", type=str, default=None, help="Arquivo CSV manifest no diretório de validação")
    parser.add_argument("--test-manifest", type=str, default=None, help="Arquivo CSV manifest no diretório de teste")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fração do conjunto de treino usada como validação quando val-root não é fornecido")
    parser.add_argument("--test-split", type=float, default=0.0, help="Fração do conjunto de treino usada como teste quando test-root não é fornecido")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=4, help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Taxa de aprendizado")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", help="Diretório para salvar checkpoints")
    parser.add_argument("--checkpoint-frequency", type=int, default=1, help="Salvar checkpoint a cada N épocas")
    parser.add_argument("--patience", type=int, default=5, help="Pacote de early stopping com base na validação")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória para reprodutibilidade")
    parser.add_argument("--experiment-name", type=str, default="medvision_experiment", help="Nome do experimento")
    parser.add_argument("--log-csv", type=str, default="training_log.csv", help="Nome do arquivo CSV de logs no diretório de checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    train_images, train_masks = build_dataset_from_root(args.train_root, manifest_file=args.train_manifest)
    val_images, val_masks = (None, None)
    test_images, test_masks = (None, None)

    if args.val_root:
        val_images, val_masks = build_dataset_from_root(args.val_root, manifest_file=args.val_manifest)

    if args.test_root:
        test_images, test_masks = build_dataset_from_root(args.test_root, manifest_file=args.test_manifest)

    if not args.val_root and len(train_images) > 1:
        if args.test_split > 0.0:
            train_images, train_masks, val_images, val_masks, test_images, test_masks = split_dataset(
                train_images,
                train_masks,
                val_fraction=args.val_split,
                test_fraction=args.test_split,
                seed=42,
            )
        else:
            train_images, train_masks, val_images, val_masks = split_dataset(
                train_images,
                train_masks,
                val_fraction=args.val_split,
                seed=42,
            )

    print(f"Treinando com {len(train_images)} imagens de treino")
    if val_images is not None:
        print(f"Validação com {len(val_images)} imagens")
    if test_images is not None and len(test_images) > 0:
        print(f"Teste com {len(test_images)} imagens")

    train_segmentation(
        train_images,
        train_masks,
        val_images,
        val_masks,
        test_images,
        test_masks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_frequency,
        patience=args.patience,
        log_csv=args.log_csv,
        seed=args.seed,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
