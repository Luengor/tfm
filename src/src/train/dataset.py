import os
import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Strong augmentation: each crop yields a different view per __getitem__,
# so multiple draws of the same image inside a PK batch act as positive variants.
# Strong enough that singleton-class anchors can form valid positive pairs from
# two augmented views of the same crop (SimCLR-style).
_AUG_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    _NORMALIZE,
])

# Style-oriented augmentation: keeps vertical flip and gaussian blur (style is
# orientation- and sharpness-invariant for graffiti) on top of the same strong
# augmentation policy used for author training.
_STYLE_AUG_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    _NORMALIZE,
])

_BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _NORMALIZE,
])


class GraffitiLabeledDataset(Dataset):
    """
    Labeled crop dataset for metric learning. Each item is (image, class_idx).
    Classes with fewer than `min_samples` images are dropped so every batch can
    form at least one positive pair after PK sampling.
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        min_samples: int = 1,
        two_views: bool = False,
        skip: set[str] | None = None,
    ):
        self.root_dir = root_dir
        self.transform = transform or _AUG_TRANSFORM
        self.two_views = two_views
        skip = skip or set()

        all_dirs = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and d not in skip
        )
        self.classes = [
            d for d in all_dirs
            if len(os.listdir(os.path.join(root_dir, d))) >= min_samples
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[str, int]] = []
        for c in self.classes:
            for img in os.listdir(os.path.join(root_dir, c)):
                self.samples.append((os.path.join(root_dir, c, img), self.class_to_idx[c]))

        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, (_, label) in enumerate(self.samples):
            self.label_to_indices[label].append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.two_views:
            return self.transform(img), self.transform(img), label
        return self.transform(img), label


class PKSampler(Sampler):
    """
    Yields batches of P classes × K samples per class.

    If a class has fewer than K samples it is sampled with replacement
    (augmentation still produces distinct views).
    """

    def __init__(self, label_to_indices: dict[int, list[int]], P: int, K: int, num_batches: int):
        self.label_to_indices = label_to_indices
        self.P = P
        self.K = K
        self.num_batches = num_batches
        self.classes = list(label_to_indices.keys())
        if len(self.classes) == 0:
            raise ValueError("PKSampler got empty label_to_indices")

    def __iter__(self):
        for _ in range(self.num_batches):
            if len(self.classes) >= self.P:
                chosen = random.sample(self.classes, self.P)
            else:
                chosen = random.choices(self.classes, k=self.P)
            batch = []
            for c in chosen:
                pool = self.label_to_indices[c]
                if len(pool) >= self.K:
                    batch.extend(random.sample(pool, self.K))
                else:
                    batch.extend(random.choices(pool, k=self.K))
            yield batch

    def __len__(self):
        return self.num_batches
