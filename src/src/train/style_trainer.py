import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict


class StyleDataset(Dataset):
    """Reads images from dataset_cropped/{class_name}/ subdirectories.
    Returns two independently augmented views of each image for SupCon.
    """

    def __init__(self, root_dir, transform=None, skip: set[str] | None = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of (path, label_idx)
        self.class_to_indices = defaultdict(list)

        skip = skip or set()
        available = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d not in skip
        )
        self.classes = available
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    idx = len(self.samples)
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))
                    self.class_to_indices[self.class_to_idx[cls]].append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            view1 = self.transform(img)
            view2 = self.transform(img)
        else:
            view1 = view2 = img
        return view1, view2, label


class BalancedBatchSampler(Sampler):
    """Yields batches with exactly `samples_per_class` images per class.
    Guarantees every anchor has at least one positive in each batch.
    """

    def __init__(self, class_to_indices: dict, samples_per_class: int, n_batches: int):
        self.class_to_indices = class_to_indices
        self.samples_per_class = samples_per_class
        self.n_batches = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for indices in self.class_to_indices.values():
                chosen = random.choices(indices, k=self.samples_per_class)
                batch.extend(chosen)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).
    Expects L2-normalised embeddings.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: (N, dim)  — must be unit-normalised before calling
        # labels:   (N,)
        device = features.device
        N = features.shape[0]

        sim = torch.mm(features, features.T) / self.temperature  # (N, N)

        # Positive mask: same label, different index
        labels_col = labels.view(-1, 1)
        pos_mask = (labels_col == labels_col.T).float()
        pos_mask.fill_diagonal_(0.0)

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Zero out self-similarity in denominator
        eye = torch.eye(N, device=device, dtype=torch.bool)
        exp_sim = torch.exp(sim).masked_fill(eye, 0.0)

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average over positive pairs; skip anchors with no in-batch positive
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        per_sample = -(pos_mask * log_prob).sum(dim=1)
        per_sample[valid] = per_sample[valid] / pos_count[valid]

        return per_sample[valid].mean() if valid.any() else per_sample.mean()


def _build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_style_training(
    model: nn.Module,
    save_path: str,
    dataset_root: str = "../dataset_cropped",
    num_epochs: int = 60,
    samples_per_class: int = 4,
    learning_rate: float = 1e-4,
    temperature: float = 0.07,
    patience: int = 15,
    skip: set[str] | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = StyleDataset(root_dir=dataset_root, transform=_build_transform(), skip=skip)
    n_classes = len(dataset.classes)
    print(f"Classes ({n_classes}): {dataset.classes}")
    print(f"Samples per class: { {c: len(v) for c, v in dataset.class_to_indices.items()} }")

    # ~3–4 passes over the smallest class per epoch
    n_batches = max(min(len(dataset) // (n_classes * samples_per_class), 30), 10)
    sampler = BalancedBatchSampler(dataset.class_to_indices, samples_per_class, n_batches)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=2, pin_memory=True)

    model = model.to(device)
    criterion = SupConLoss(temperature=temperature)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for view1, view2, labels in pbar:
            # Stack both augmented views into a single forward pass
            images = torch.cat([view1, view2], dim=0).to(device)       # (2N, C, H, W)
            labels_2x = torch.cat([labels, labels], dim=0).to(device)  # (2N,)

            optimizer.zero_grad()
            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            loss = criterion(embeddings, labels_2x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.4f}"})

        scheduler.step()
        avg_loss = running_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    print(f"Training complete. Best model saved to {save_path} (loss={best_loss:.4f})")
