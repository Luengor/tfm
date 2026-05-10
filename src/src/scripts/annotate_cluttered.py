import os
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from tqdm import tqdm
import argparse

def annotate_cluttered(annotations_xml, image_dir, model_path, output_xml):
    """
    Annotate images in a CVAT XML that have no bounding boxes or are tagged as 'cluttered'.
    Replaces any existing boxes in those images with new detections from a YOLO model,
    all labeled as 'other'.
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Parsing annotations from {annotations_xml}...")
    tree = ET.parse(annotations_xml)
    root = tree.getroot()
    
    # Identify images meeting the criteria
    images_to_process = []
    for image in root.findall('image'):
        boxes = image.findall('box')
        tags = image.findall('tag')
        
        has_boxes = len(boxes) > 0
        is_cluttered = any(tag.get('label') == 'cluttered' for tag in tags)
        
        if not has_boxes or is_cluttered:
            images_to_process.append((image, boxes))
            
    print(f"Found {len(images_to_process)} images to (re)annotate out of {len(root.findall('image'))}.")
    
    for image, existing_boxes in tqdm(images_to_process):
        filename = image.get('name')
        img_path = os.path.join(image_dir, filename)
        
        if not os.path.exists(img_path):
            tqdm.write(f"Warning: Image {img_path} not found. Skipping.")
            continue
            
        # Remove existing boxes as requested
        for box in existing_boxes:
            image.remove(box)
            
        # Run YOLO inference
        results = model(img_path, verbose=False)
        
        # Add new boxes with label 'other'
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].tolist() # [xtl, ytl, xbr, ybr]
                
                # Create the box element
                ET.SubElement(image, "box", {
                    "label": "other",
                    "source": "file",
                    "occluded": "0",
                    "xtl": f"{coords[0]:.2f}",
                    "ytl": f"{coords[1]:.2f}",
                    "xbr": f"{coords[2]:.2f}",
                    "ybr": f"{coords[3]:.2f}",
                    "z_order": "0"
                })

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_xml))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Format the XML (available in Python 3.9+)
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="  ", level=0)
    
    print(f"Saving updated annotations to {output_xml}...")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate images without boxes or with 'cluttered' tag using YOLO.")
    parser.add_argument("--annotations", default="grafiti/annotations.xml", help="Path to input annotations.xml")
    parser.add_argument("--image_dir", default="grafiti/images/train", help="Directory containing images")
    parser.add_argument("--model", required=True, help="Path to YOLO model weights")
    parser.add_argument("--output", default="grafiti/annotations_updated.xml", help="Path to output XML")
    
    args = parser.parse_args()
    
    annotate_cluttered(args.annotations, args.image_dir, args.model, args.output)
