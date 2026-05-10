import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

def crop_images(annotations_path, images_dir, output_dir, padding=0.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tree = ET.parse(annotations_path)
    root = tree.getroot()

    for image_tag in tqdm(root.findall('image'), desc="Processing images"):
        image_name = image_tag.get('name')
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found.")
            continue

        boxes = image_tag.findall('box')
        if not boxes:
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                for i, box in enumerate(boxes):
                    label = box.get('label')
                    xtl = float(box.get('xtl'))
                    ytl = float(box.get('ytl'))
                    xbr = float(box.get('xbr'))
                    ybr = float(box.get('ybr'))

                    if padding > 0:
                        box_w = xbr - xtl
                        box_h = ybr - ytl
                        xtl = max(0, xtl - box_w * padding)
                        ytl = max(0, ytl - box_h * padding)
                        xbr = min(width, xbr + box_w * padding)
                        ybr = min(height, ybr + box_h * padding)

                    # Create label directory
                    label_dir = os.path.join(output_dir, label)
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir)

                    # Crop and save
                    crop = img.crop((xtl, ytl, xbr, ybr))
                    
                    # Convert to RGB if necessary (e.g. for RGBA images)
                    if crop.mode != 'RGB':
                        crop = crop.convert('RGB')
                        
                    crop_filename = f"{os.path.splitext(image_name)[0]}_crop_{i}.jpg"
                    crop.save(os.path.join(label_dir, crop_filename))
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")

if __name__ == "__main__":
    ANNOTATIONS_PATH = os.path.join("dataset", "annotations.xml")
    IMAGES_DIR = os.path.join("dataset", "images")
    OUTPUT_DIR = os.path.join("dataset", "crops")
    PADDING = 0.1 # 10% padding
    
    crop_images(ANNOTATIONS_PATH, IMAGES_DIR, OUTPUT_DIR, padding=PADDING)
