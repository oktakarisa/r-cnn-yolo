#!/usr/bin/env python3
"""
Script to prepare Simpsons dataset for YOLOv3 training
Converts from Faster R-CNN format to YOLOv3 format
"""

import os
import sys
import shutil
from pathlib import Path
import argparse

def read_faster_rcnn_annotations(annotation_file):
    """Read Faster R-CNN annotation format and extract classes"""
    classes = set()
    annotations = []
    
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 6:
                continue
                
            image_path = parts[0]
            # Format: x_min,y_min,x_max,y_max,class_name
            bbox_info = parts[1:6]
            class_name = bbox_info[-1]
            
            # Handle multiple bounding boxes in same line
            remaining_parts = parts[6:]
            while len(remaining_parts) >= 4:
                # Additional bbox: x_min,y_min,x_max,y_max,class_name
                class_name = remaining_parts[-1]
                remaining_parts = remaining_parts[5:]
            
            classes.add(class_name)
            annotations.append(line)
    
    return sorted(list(classes)), annotations

def create_yolov3_class_file(classes, output_path):
    """Create YOLOv3 classes.txt file"""
    with open(output_path, 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    print(f"Created classes file: {output_path}")

def convert_faster_rcnn_to_yolo(faster_rcnn_file, output_dir, image_dir):
    """Convert Faster R-CNN annotations to YOLOv3 format"""
    # Create output directories
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    # Read annotations and extract classes
    classes, annotations = read_faster_rcnn_annotations(faster_rcnn_file)
    
    # Create classes file
    create_yolov3_class_file(classes, f"{output_dir}/classes.txt")
    
    # Create class to index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    processed_images = set()
    
    for annotation in annotations:
        parts = annotation.split(',')
        if len(parts) < 6:
            continue
            
        image_path = parts[0]
        image_name = os.path.basename(image_path)
        
        # Skip if already processed this image
        if image_name in processed_images:
            continue
            
        processed_images.add(image_name)
        
        # Copy image to output directory if it exists
        source_image_path = os.path.join(image_dir, image_path)
        target_image_path = f"{output_dir}/images/{image_name}"
        
        if os.path.exists(source_image_path):
            shutil.copy2(source_image_path, target_image_path)
            print(f"Copied image: {image_name}")
        else:
            print(f"Warning: Image not found: {source_image_path}")
            continue
        
        # Process all bounding boxes for this image
        current_parts = parts[1:]
        yolo_annotations = []
        
        while len(current_parts) >= 5:
            try:
                x_min = float(current_parts[0])
                y_min = float(current_parts[1])
                x_max = float(current_parts[2])
                y_max = float(current_parts[3])
                class_name = current_parts[4]
                
                # Get image dimensions (would need to read actual image)
                # For now, assuming standard dimensions - this would need to be fixed
                img_width = 224  # Placeholder - should read from actual image
                img_height = 224  # Placeholder - should read from actual image
                
                # Convert to YOLO format (normalized center coordinates + width/height)
                x_center = (x_min + x_max) / 2.0 / img_width
                y_center = (y_min + y_max) / 2.0 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Clamp values to [0,1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                class_idx = class_to_idx[class_name]
                yolo_annotations.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Move to next bbox if any
                current_parts = current_parts[5:] if len(current_parts) > 5 else []
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing annotation: {e}")
                break
        
        # Save YOLO format annotations
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = f"{output_dir}/labels/{label_name}"
        
        with open(label_path, 'w') as f:
            for ann in yolo_annotations:
                f.write(ann + '\n')
    
    return len(processed_images)

def create_train_val_split(dataset_dir, train_ratio=0.8):
    """Create train.txt and val.txt files for YOLOv3 training"""
    import random
    
    images_dir = f"{dataset_dir}/images"
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Create train.txt
    with open(f"{dataset_dir}/train.txt", 'w') as f:
        for img_file in train_files:
            img_path = os.path.abspath(f"{images_dir}/{img_file}")
            f.write(img_path + '\n')
    
    # Create val.txt
    with open(f"{dataset_dir}/val.txt", 'w') as f:
        for img_file in val_files:
            img_path = os.path.abspath(f"{images_dir}/{img_file}")
            f.write(img_path + '\n')
    
    print(f"Created train/val split: {len(train_files)} train, {len(val_files)} validation")

def main():
    parser = argparse.ArgumentParser(description="Convert Simpsons dataset for YOLOv3 training")
    parser.add_argument("--input", default="data/faster_rcnn/annotation.txt",
                       help="Path to Faster R-CNN annotation file")
    parser.add_argument("--image_dir", default="data/faster_rcnn/simpsons_dataset",
                       help="Path to Simpsons dataset images")
    parser.add_argument("--output", default="data/yolov3/simpsons",
                       help="Output directory for YOLOv3 format data")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Training/validation split ratio")
    
    args = parser.parse_args()
    
    print("="*50)
    print("Simpsons Dataset Conversion for YOLOv3")
    print("="*50)
    
    if not os.path.exists(args.input):
        print(f"Error: Annotation file not found: {args.input}")
        return
    
    print(f"Input annotation file: {args.input}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output}")
    
    # Convert dataset
    num_images = convert_faster_rcnn_to_yolo(args.input, args.output, args.image_dir)
    print(f"\nProcessed {num_images} images")
    
    # Create train/val split
    create_train_val_split(args.output, args.train_ratio)
    
    print("\nDataset conversion completed!")
    print(f"Output directory structure:")
    print(f"  {args.output}/")
    print(f"  ├── images/      # Training images")
    print(f"  ├── labels/      # YOLO format labels")
    print(f"  ├── classes.txt  # Class names")
    print(f"  ├── train.txt    # Training image list")
    print(f"  └── val.txt      # Validation image list")

if __name__ == "__main__":
    main()
