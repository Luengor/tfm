import os
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image

def create_cvat_xml(image_dir, model_path, output_xml):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    class_names = model.names
    
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    
    meta = ET.SubElement(root, "meta")
    # Minimal meta info
    job = ET.SubElement(meta, "job")
    ET.SubElement(job, "id").text = "0"
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    ET.SubElement(job, "size").text = str(len(image_files))
    ET.SubElement(job, "mode").text = "annotation"
    
    labels = ET.SubElement(meta, "labels")
    for idx, name in class_names.items():
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = name
        ET.SubElement(label, "color").text = "#00ff00"
        ET.SubElement(label, "type").text = "rectangle"

    print(f"Processing {len(image_files)} images from {image_dir}...")
    
    for i, filename in enumerate(tqdm(image_files)):
        img_path = os.path.join(image_dir, filename)
        
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue
            
        results = model(img_path, verbose=False)
        
        image_elem = ET.SubElement(root, "image", id=str(i), name=filename, width=str(width), height=str(height))
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = box.xyxy[0].tolist() # [xtl, ytl, xbr, ybr]
                cls_id = int(box.cls[0])
                label_name = class_names[cls_id]
                
                ET.SubElement(image_elem, "box", {
                    "label": label_name,
                    "source": "manual",
                    "occluded": "0",
                    "xtl": f"{coords[0]:.2f}",
                    "ytl": f"{coords[1]:.2f}",
                    "xbr": f"{coords[2]:.2f}",
                    "ybr": f"{coords[3]:.2f}",
                    "z_order": "0"
                })

    tree = ET.ElementTree(root)
    if hasattr(ET, 'indent'):
        ET.indent(tree, space="  ", level=0)
    
    print(f"Saving annotations to {output_xml}...")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-annotate images using YOLO and export to CVAT 1.1 format.")
    parser.add_argument("--image_dir", default="stopgrafiti", help="Directory containing images")
    parser.add_argument("--model_path", default="train-3/weights/best.pt", help="Path to YOLO model weights")
    parser.add_argument("--output_xml", default="stopgrafiti_annotations.xml", help="Output XML filename")
    
    args = parser.parse_args()
    create_cvat_xml(args.image_dir, args.model_path, args.output_xml)
