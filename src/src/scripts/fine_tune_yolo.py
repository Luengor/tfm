import os
import shutil
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO

def get_cluttered_images(xml_path):
    """
    Parses the annotations.xml to find images tagged as 'cluttered'.
    Returns a set of image filenames.
    """
    if not xml_path.exists():
        print(f"Warning: {xml_path} not found. No images will be marked as cluttered.")
        return set()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cluttered_images = set()

    for image in root.findall('image'):
        image_name = image.get('name')
        for tag in image.findall('tag'):
            if tag.get('label') == 'cluttered':
                cluttered_images.add(image_name)
                break
    return cluttered_images

def preprocess_dataset(input_dir, output_dir, target_class_name="graffiti", dry_run=False):
    """
    Preprocesses the dataset:
    - Prepares labels and images for training.
    - Filters out images without bounding boxes.
    - Uses annotations.xml to separate 'cluttered' images for evaluation.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    xml_path = input_path / "annotations.xml"

    # Train split (clean images)
    train_img_out = output_path / "images" / "train"
    train_lb_out = output_path / "labels" / "train"
    # Val split (cluttered images)
    val_img_out = output_path / "images" / "val"
    val_lb_out = output_path / "labels" / "val"

    if not dry_run:
        for d in [train_img_out, train_lb_out, val_img_out, val_lb_out]:
            d.mkdir(parents=True, exist_ok=True)

    cluttered_images = get_cluttered_images(xml_path)
    if dry_run:
        print(f"[Dry Run] Found {len(cluttered_images)} cluttered images in XML.")
    else:
        print(f"Found {len(cluttered_images)} cluttered images in XML.")

    label_files = list((input_path / "labels" / "train").glob("*.txt"))
    train_count = 0
    val_count = 0
    skipped_empty = 0
    skipped_no_img = 0

    for lb_file in label_files:
        img_name_stem = lb_file.stem

        # Find corresponding image
        img_file = None
        img_full_name = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            candidate = input_path / "images" / "train" / f"{img_name_stem}{ext}"
            if candidate.exists():
                img_file = candidate
                img_full_name = f"{img_name_stem}{ext}"
                break

        if not img_file:
            skipped_no_img += 1
            continue

        with open(lb_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            skipped_empty += 1
            continue

        new_labels = [line.strip() for line in lines if line.strip()]

        # Decide split
        if img_full_name in cluttered_images:
            img_dest = val_img_out
            lb_dest = val_lb_out
            val_count += 1
        else:
            img_dest = train_img_out
            lb_dest = train_lb_out
            train_count += 1

        if not dry_run:
            shutil.copy(img_file, img_dest / img_file.name)
            with open(lb_dest / f"{img_name_stem}.txt", 'w') as f:
                f.write('\n'.join(new_labels) + '\n')

    print(f"Preprocessing results {'(Dry Run)' if dry_run else ''}:")
    print(f"- Training images (clean): {train_count}")
    print(f"- Validation images (cluttered): {val_count}")
    print(f"- Skipped empty: {skipped_empty}")
    print(f"- Skipped (no image found): {skipped_no_img}")

    if dry_run:
        return None

    # Try to get all class names from original data.yaml to keep indices consistent
    names = {0: target_class_name}
    orig_data_yaml = input_path / "data.yaml"
    if orig_data_yaml.exists():
        with open(orig_data_yaml, 'r') as f:
            old_data = yaml.safe_load(f)
            old_names = old_data.get('names', {})
            if isinstance(old_names, dict):
                names = {i: target_class_name for i in old_names.keys()}
            elif isinstance(old_names, list):
                names = {i: target_class_name for i in range(len(old_names))}

    # Create new data.yaml
    new_data_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': names
    }

    with open(output_path / "data.yaml", 'w') as f:
        yaml.dump(new_data_config, f)

    return output_path / "data.yaml"

def train_yolo(data_yaml, model_name="yolov8n.pt", epochs=50, imgsz=640, batch=16, device=None):
    if data_yaml is None:
        print("[Dry Run] Skipping training.")
        return None
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=True,
        single_cls=True
    )
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on graffiti dataset")
    parser.add_argument("--dataset", type=str, default="dataset", help="Input dataset directory")
    parser.add_argument("--output", type=str, default="processed_dataset", help="Output directory for processed dataset")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16). Reduce if OOM occurs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 0, 0,1, cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without copying files or training")

    args = parser.parse_args()

    data_yaml_path = preprocess_dataset(args.dataset, args.output, dry_run=args.dry_run)
    train_yolo(data_yaml_path, model_name=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device)
