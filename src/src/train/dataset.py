import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Strong augmentation for positive pairs — used for both anchor and positive
# to ensure they're meaningfully different views of the same author's work.
_AUG_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    _NORMALIZE,
])

_BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _NORMALIZE,
])


class GraffitiTripletDataset(Dataset):
    """
    Triplet dataset for graffiti author metric learning.

    Anchor classes: authors with ≥2 crops. Positive pairs are formed from two
    different real images of the same author, each passed through the strong
    augmentation pipeline.

    Singleton classes (exactly 1 crop) are excluded as anchors but used as
    hard negatives. When sampling a negative, `hard_negative_prob` controls
    how often the negative is drawn from the singleton pool vs. the full pool.
    """

    def __init__(self, root_dir, transform=None, hard_negative_prob: float = 0.5):
        self.root_dir = root_dir
        self.transform = transform or _AUG_TRANSFORM
        self.hard_negative_prob = hard_negative_prob

        all_dirs = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.class_to_images = {
            cls: [os.path.join(root_dir, cls, img) for img in os.listdir(os.path.join(root_dir, cls))]
            for cls in all_dirs
        }

        # Only classes with ≥2 real images can form valid positive pairs
        self.anchor_classes = [cls for cls in all_dirs if len(self.class_to_images[cls]) >= 2]
        # Singletons used exclusively as hard negatives
        self.singleton_classes = [cls for cls in all_dirs if len(self.class_to_images[cls]) == 1]
        # Full negative pool (includes both multi-sample and singleton classes)
        self.all_classes = all_dirs

        self.all_images = [
            (img_path, cls)
            for cls in self.anchor_classes
            for img_path in self.class_to_images[cls]
        ]

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_path, anchor_class = self.all_images[idx]

        # Positive: a different real image from the same author
        positives = [p for p in self.class_to_images[anchor_class] if p != anchor_path]
        positive_path = random.choice(positives)

        # Negative: preferentially drawn from singletons (hard negatives)
        singleton_pool = [c for c in self.singleton_classes if c != anchor_class]
        if singleton_pool and random.random() < self.hard_negative_prob:
            neg_class = random.choice(singleton_pool)
        else:
            neg_class = random.choice([c for c in self.all_classes if c != anchor_class])
        negative_path = random.choice(self.class_to_images[neg_class])

        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')

        return (
            self.transform(anchor_img),
            self.transform(positive_img),
            _BASE_TRANSFORM(negative_img),
        )
