import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.train.dataset import GraffitiLabeledDataset, PKSampler, _BASE_TRANSFORM
from src.train.evaluate import compute_metrics, list_eligible_classes
from src.train.losses import SupConLoss, batch_hard_triplet_loss


@torch.no_grad()
def _extract_val_embeddings(model, val_dir: str, device, min_samples: int = 2):
    classes, skipped = list_eligible_classes(val_dir, min_samples)
    embeddings = []
    labels = []
    for label_idx, cls in enumerate(classes):
        cls_dir = os.path.join(val_dir, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                x = _BASE_TRANSFORM(img).unsqueeze(0).to(device)
                emb = model(x).squeeze(0).cpu().numpy()
                embeddings.append(emb)
                labels.append(label_idx)
            except Exception as e:
                print(f"  Val error {img_path}: {e}")
    return np.array(embeddings), np.array(labels), len(classes), skipped


def run_training(
    model,
    save_path,
    dataset_dir: str = os.path.join("dataset", "crops"),
    num_epochs: int = 20,
    classes_per_batch: int = 8,
    samples_per_class: int = 4,
    batches_per_epoch: int | None = None,
    learning_rate: float = 1e-4,
    loss: str = "supcon",
    margin: float = 0.3,
    temperature: float = 0.07,
    train_min_samples: int = 1,
    val_dataset_dir: str | None = None,
    val_min_samples: int = 2,
    val_metric: str = "accuracy_1nn",
    patience: int | None = None,
):
    if loss not in {"supcon", "triplet"}:
        raise ValueError(f"Unknown loss '{loss}'. Choose 'supcon' or 'triplet'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loss: {loss}")

    two_views = loss == "supcon"
    dataset = GraffitiLabeledDataset(
        root_dir=dataset_dir, min_samples=train_min_samples, two_views=two_views
    )
    print(
        f"Dataset: {len(dataset)} images across {len(dataset.classes)} classes "
        f"(≥{train_min_samples} samples each)"
    )
    if len(dataset.classes) < 2:
        raise ValueError("Need at least 2 classes for metric-learning training")

    if batches_per_epoch is None:
        batches_per_epoch = max(1, len(dataset) // (classes_per_batch * samples_per_class))
    print(
        f"PK batches: P={classes_per_batch} K={samples_per_class} "
        f"batch_size={classes_per_batch * samples_per_class}"
        f"{' (×2 views)' if two_views else ''} batches/epoch={batches_per_epoch}"
    )

    sampler = PKSampler(dataset.label_to_indices, classes_per_batch, samples_per_class, batches_per_epoch)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    if val_dataset_dir:
        val_classes, val_skipped = list_eligible_classes(val_dataset_dir, val_min_samples)
        print(
            f"Validation: {val_dataset_dir} — {len(val_classes)} classes with "
            f"≥{val_min_samples} samples ({val_skipped} skipped)"
        )

    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    supcon = SupConLoss(temperature=temperature) if loss == "supcon" else None

    best_score = float("-inf")
    patience_counter = 0
    best_epoch = -1
    saved_best = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            if two_views:
                v1, v2, lbls = batch
                imgs = torch.cat([v1, v2], dim=0).to(device)
                lbls = torch.cat([lbls, lbls], dim=0).to(device)
                emb = model(imgs)
                emb = F.normalize(emb, p=2, dim=1)
                step_loss = supcon(emb, lbls)
            else:
                imgs, lbls = batch
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                emb = model(imgs)
                emb = F.normalize(emb, p=2, dim=1)
                step_loss = batch_hard_triplet_loss(emb, lbls, margin)

            step_loss.backward()
            optimizer.step()

            running_loss += step_loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

        scheduler.step()

        if val_dataset_dir:
            model.eval()
            embs, lbls_np, n_classes, _ = _extract_val_embeddings(
                model, val_dataset_dir, device, min_samples=val_min_samples
            )
            if len(embs) > 0 and n_classes >= 2:
                metrics = compute_metrics(embs, lbls_np, n_classes)
                print(
                    f"  Val epoch {epoch + 1}: silhouette={metrics['silhouette']:.4f} "
                    f"acc_1nn={metrics['accuracy_1nn']:.4f} "
                    f"ari={metrics['ari']:.4f} nmi={metrics['nmi']:.4f}"
                )
                if val_metric not in metrics:
                    raise ValueError(f"Unknown val_metric '{val_metric}'. Available: {list(metrics)}")
                score = metrics[val_metric]
                if score > best_score:
                    best_score = score
                    best_epoch = epoch + 1
                    patience_counter = 0
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    saved_best = True
                    print(f"  New best {val_metric}={score:.4f} — saved to {save_path}")
                else:
                    patience_counter += 1
                    if patience is not None and patience_counter >= patience:
                        print(
                            f"Early stopping at epoch {epoch + 1} "
                            f"(no improvement in {val_metric} for {patience} epochs)"
                        )
                        break
            else:
                print(f"  Val epoch {epoch + 1}: not enough samples/classes to compute metrics")

    if val_dataset_dir and saved_best:
        print(
            f"Training complete. Best {val_metric}={best_score:.4f} at epoch {best_epoch}; "
            f"best weights saved to {save_path}"
        )
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
