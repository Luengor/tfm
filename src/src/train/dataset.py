import os
import random
from PIL import Image
from torch.utils.data import Dataset

class GraffitiTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        all_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_images = {cls: [os.path.join(root_dir, cls, img) for img in os.listdir(os.path.join(root_dir, cls))] for cls in all_dirs}
        
        # Anchor classes: must have at least 2 images and NOT be 'other'
        self.anchor_classes = [cls for cls in all_dirs if len(self.class_to_images[cls]) >= 2 and cls.lower() != 'other']
        # Negative pool: all available classes
        self.negative_pool_classes = all_dirs
        
        self.all_images = []
        for cls in self.anchor_classes:
            for img_path in self.class_to_images[cls]:
                self.all_images.append((img_path, cls))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        anchor_path, anchor_class = self.all_images[idx]
        
        # Positive: same class, different image
        # (Guaranteed to have at least 2 images due to anchor_classes filter)
        positive_path = random.choice(self.class_to_images[anchor_class])
        while positive_path == anchor_path:
            positive_path = random.choice(self.class_to_images[anchor_class])
            
        # Negative: different class (can be 'other')
        negative_class = random.choice([cls for cls in self.negative_pool_classes if cls != anchor_class])
        negative_path = random.choice(self.class_to_images[negative_class])
        
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(positive_path).convert('RGB')
        negative_img = Image.open(negative_path).convert('RGB')
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            
        return anchor_img, positive_img, negative_img
