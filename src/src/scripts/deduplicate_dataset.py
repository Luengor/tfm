import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sys

# Add project root to sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.embedding.embeddings import EmbeddingModelNames, get_model

def get_image_fingerprint(image_path: Path, model) -> np.ndarray:
    """Generate an embedding for the image to use as a fingerprint."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        embedding = model.gen_embedding(img)
    return np.array(embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def parse_cvat_annotations(xml_path: Path) -> Dict[str, List[ET.Element]]:
    """Parse CVAT annotations.xml and return a mapping of filename to its image element."""
    if not xml_path.exists():
        return {}
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = {}
    for img_tag in root.findall("image"):
        name = img_tag.get("name")
        # Store the entire element so we can copy it later
        annotations[name] = img_tag
        
    return annotations

def count_annotations(image_element: ET.Element) -> int:
    """Count total annotations (boxes, polygons, etc.) in a CVAT image element."""
    if image_element is None:
        return 0
    # Common CVAT annotation tags
    tags = ["box", "polygon", "polyline", "points", "ellipse"]
    count = 0
    for tag in tags:
        count += len(image_element.findall(tag))
    return count

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def draw_annotations(image: Image.Image, image_element: ET.Element) -> Image.Image:
    """Draw CVAT annotations on a copy of the image."""
    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    
    if image_element is None:
        return img_copy
        
    for box in image_element.findall("box"):
        xtl = float(box.get("xtl", 0))
        ytl = float(box.get("ytl", 0))
        xbr = float(box.get("xbr", 0))
        ybr = float(box.get("ybr", 0))
        label = box.get("label", "")
        
        draw.rectangle([xtl, ytl, xbr, ybr], outline="red", width=3)
        draw.text((xtl, ytl), label, fill="red")
        
    return img_copy

def visualize_group(image_paths: List[Path], annotations: Dict[str, ET.Element]):
    """Show a group of duplicate images side-by-side with annotations."""
    n = len(image_paths)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
        
    for i, path in enumerate(image_paths):
        with Image.open(path) as img:
            elem = annotations.get(path.name)
            annotated_img = draw_annotations(img, elem)
            axes[i].imshow(annotated_img)
            count = count_annotations(elem)
            axes[i].set_title(f"[{i}] {path.name}\n({count} annotations)")
            axes[i].axis("off")
            
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

def main():
    parser = argparse.ArgumentParser(description="De-duplicate a CVAT dataset based on visual similarity.")
    parser.add_argument("--input", required=True, help="Input dataset directory (containing annotations.xml and images/)")
    parser.add_argument("--output", required=True, help="Output directory for the cleaned dataset")
    parser.add_argument("--threshold", type=float, default=0.98, help="Cosine similarity threshold for de-duplication (default: 0.98)")
    parser.add_argument("--model", default="dinov2_vits14", choices=[m.value for m in EmbeddingModelNames], 
                        help="Embedding model to use for similarity (default: dinov2_vits14)")
    parser.add_argument("--interactive", action="store_true", help="Ask for confirmation on ambiguous cases")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    images_dir = input_dir / "images"
    xml_path = input_dir / "annotations.xml"

    if not images_dir.exists():
        print(f"Error: images/ directory not found in {args.input}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)

    print(f"Loading model: {args.model}...")
    model = get_model(EmbeddingModelNames(args.model))
    
    print("Parsing annotations...")
    annotations = parse_cvat_annotations(xml_path)
    
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    print(f"Found {len(image_paths)} images.")

    print("Generating fingerprints...")
    fingerprints = {}
    for p in tqdm(image_paths, desc="Fingerprinting"):
        try:
            fingerprints[p.name] = get_image_fingerprint(p, model)
        except Exception as e:
            print(f"Error processing {p.name}: {e}")

    to_keep: Set[str] = set()
    to_remove: Set[str] = set()
    
    # Process images one by one
    processed_names = sorted(fingerprints.keys())
    
    print("Finding duplicates...")
    for i, name_a in enumerate(tqdm(processed_names, desc="De-duplicating")):
        if name_a in to_remove or name_a in to_keep:
            continue
            
        fp_a = fingerprints[name_a]
        current_group = [name_a]
        
        # Check against all subsequent images
        for j in range(i + 1, len(processed_names)):
            name_b = processed_names[j]
            if name_b in to_remove or name_b in to_keep:
                continue
                
            fp_b = fingerprints[name_b]
            sim = cosine_similarity(fp_a, fp_b)
            
            if sim >= args.threshold:
                current_group.append(name_b)
        
        if len(current_group) == 1:
            to_keep.add(name_a)
            continue
            
        # Handle group of duplicates
        print(f"\nFound duplicate group: {current_group}")
        
        # Sort by annotation count (descending)
        group_with_counts = []
        for name in current_group:
            elem = annotations.get(name)
            count = count_annotations(elem) if elem is not None else 0
            group_with_counts.append((name, count))
            
        group_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        winner_name, winner_count = group_with_counts[0]
        others = group_with_counts[1:]
        
        # Decision logic
        final_winner = winner_name
        
        # Check if there's a tie in counts > 0
        potential_conflicts = [x for x in others if x[1] == winner_count and x[1] > 0]
        
        if potential_conflicts and args.interactive:
            print(f"Conflict: Multiple images have {winner_count} annotations.")
            
            # Paths for visualization
            group_paths = [images_dir / name for name, _ in group_with_counts]
            visualize_group(group_paths, annotations)
            
            for idx, (name, count) in enumerate(group_with_counts):
                print(f"  [{idx}] {name} ({count} annotations)")
            
            while True:
                choice = input(f"Which one to keep? [0-{len(group_with_counts)-1}, s to skip]: ").strip().lower()
                if choice == 's':
                    break
                try:
                    choice_idx = int(choice)
                    if 0 <= choice_idx < len(group_with_counts):
                        final_winner = group_with_counts[choice_idx][0]
                        break
                except ValueError:
                    pass
            plt.close()
        elif args.interactive and len(group_with_counts) > 1:
            # Even if no conflict in counts, show them if interactive
            print(f"Duplicate group found. Winner by annotation count: {winner_name} ({winner_count})")
            group_paths = [images_dir / name for name, _ in group_with_counts]
            visualize_group(group_paths, annotations)
            
            choice = input(f"Keep {winner_name}? [y/n, or index 0-{len(group_with_counts)-1}]: ").strip().lower()
            if choice == 'n':
                 # Prompt for which one then
                 for idx, (name, count) in enumerate(group_with_counts):
                    print(f"  [{idx}] {name} ({count} annotations)")
                 choice_idx = int(input("Index to keep: "))
                 final_winner = group_with_counts[choice_idx][0]
            elif choice != 'y' and choice != '':
                try:
                    final_winner = group_with_counts[int(choice)][0]
                except (ValueError, IndexError):
                    pass
            plt.close()
        elif potential_conflicts:
            print(f"Tie detected for {winner_name} and {potential_conflicts[0][0]}. Keeping {winner_name} by default.")

        to_keep.add(final_winner)
        for name, _ in group_with_counts:
            if name != final_winner:
                to_remove.add(name)

    print(f"\nSummary:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  To keep:      {len(to_keep)}")
    print(f"  To remove:    {len(to_remove)}")

    # Create new XML
    if xml_path.exists():
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Create a new root and copy global metadata if possible
        new_root = ET.Element("annotations")
        # Copy version, meta, etc if they exist (simplified)
        for child in root:
            if child.tag != "image":
                # Deep copy meta/version
                new_root.append(shutil.copytree if False else child) # placeholder logic
        
        # Actually CVAT format is a bit specific, let's just filter the existing tree
        images_to_remove = []
        for img_tag in root.findall("image"):
            if img_tag.get("name") not in to_keep:
                images_to_remove.append(img_tag)
        
        for img_tag in images_to_remove:
            root.remove(img_tag)
            
        tree.write(output_dir / "annotations.xml", encoding="utf-8", xml_declaration=True)
    
    # Copy images
    for name in tqdm(to_keep, desc="Copying images"):
        src = images_dir / name
        dst = output_dir / "images" / name
        shutil.copy2(src, dst)

    print(f"Cleaned dataset saved to: {args.output}")

if __name__ == "__main__":
    main()
