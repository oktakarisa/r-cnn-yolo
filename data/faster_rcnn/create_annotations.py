#!/usr/bin/env python3
"""
Script to create proper Faster R-CNN annotations from Simpsons dataset.
This script generates annotations by detecting faces/characters in images
and creates bounding boxes in the correct format: image_path,x_min,y_min,x_max,y_max,class_name

The annotation file format:
- One line per bounding box
- Format: image_path,x_min,y_min,x_max,y_max,class_name
- Coordinates must satisfy: x_min < x_max and y_min < y_max
- Coordinates are pixel values (0-based)
- All coordinates must be within image bounds
"""

import cv2
import os
import argparse
import hashlib
from pathlib import Path
import random

def detect_character_in_image(image_path):
    """
    Detect character/face in image and return bounding box.
    Uses OpenCV's Haar Cascade for face detection as a baseline.
    Returns: (x_min, y_min, x_max, y_max) or None if detection fails
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    height, width = img.shape[:2]
    
    # Try face detection first (characters are cartoon faces)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        # Use the largest face detected
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        x_min = max(0, x)
        y_min = max(0, y)
        x_max = min(width, x + w)
        y_max = min(height, y + h)
        
        # Ensure valid bounding box
        if x_min < x_max and y_min < y_max:
            return (x_min, y_min, x_max, y_max)
    
    # Fallback: if no face detected, use full image or a reasonable default
    # For Simpsons dataset, characters often occupy most of the image
    # Use 80% of image centered
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    x_min = margin_x
    y_min = margin_y
    x_max = width - margin_x
    y_max = height - margin_y
    
    if x_min < x_max and y_min < y_max:
        return (x_min, y_min, x_max, y_max)
    
    # Last resort: use full image
    return (0, 0, width, height)

def create_annotations_from_directory(dataset_dir, output_file, train_split=0.8):
    """
    Create annotations from Simpsons dataset directory structure.
    
    Args:
        dataset_dir: Path to simpsons_dataset directory (contains character subdirectories)
        output_file: Path to output annotation file
        train_split: Ratio of images to use for training (0.8 = 80% train, 20% test)
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    annotations = []
    classes = set()
    image_count = 0
    
    print(f"Scanning dataset directory: {dataset_dir}")
    print("=" * 70)
    
    # Walk through character directories
    character_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if len(character_dirs) == 0:
        raise ValueError(f"No character directories found in {dataset_dir}")
    
    print(f"Found {len(character_dirs)} character directories")
    
    for char_dir in character_dirs:
        character_name = char_dir.name
        classes.add(character_name)
        
        # Find all image files in this character directory
        image_files = sorted([
            f for f in char_dir.iterdir() 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        print(f"Processing {character_name}: {len(image_files)} images")
        
        for img_file in image_files:
            # Create path relative to dataset directory parent (matching annotation format)
            # Format: simpsons_dataset/character_name/pic_xxxx.jpg
            rel_path = img_file.relative_to(dataset_path.parent)
            image_path = str(rel_path).replace('\\', '/')
            
            # Detect bounding box
            bbox = detect_character_in_image(str(img_file))
            
            if bbox is None:
                print(f"Warning: Could not detect bounding box in {img_file}")
                continue
            
            x_min, y_min, x_max, y_max = bbox
            
            # Validate bounding box
            if x_min >= x_max or y_min >= y_max:
                print(f"Warning: Invalid bounding box for {img_file}: ({x_min},{y_min},{x_max},{y_max})")
                continue
            
            # Create annotation line
            annotation_line = f"{image_path},{int(x_min)},{int(y_min)},{int(x_max)},{int(y_max)},{character_name}"
            annotations.append(annotation_line)
            image_count += 1
    
    # Shuffle annotations deterministically (using hash of sorted order for reproducibility)
    # This ensures consistent ordering across runs
    annotations.sort()  # Sort for deterministic ordering
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(annotations)
    
    # Write annotations to file
    print("\n" + "=" * 70)
    print(f"Writing {len(annotations)} annotations to: {output_file}")
    
    with open(output_file, 'w') as f:
        for annotation in annotations:
            f.write(annotation + '\n')
    
    print(f"Successfully created annotation file with {len(annotations)} bounding boxes")
    print(f"Found {len(classes)} unique classes:")
    for cls in sorted(classes):
        count = sum(1 for ann in annotations if ann.endswith(f",{cls}"))
        print(f"  {cls}: {count} annotations")
    
    return len(annotations), len(classes)

def fix_existing_annotations(input_file, output_file):
    """
    Fix existing annotation file by validating and correcting coordinates.
    This function reads an existing annotation file, validates all bounding boxes,
    fixes swapped coordinates, and writes a corrected version.
    """
    if not os.path.exists(input_file):
        raise ValueError(f"Input annotation file not found: {input_file}")
    
    fixed_lines = []
    stats = {
        'total': 0,
        'fixed': 0,
        'invalid': 0,
        'skipped': 0
    }
    
    print(f"Reading and fixing annotations from: {input_file}")
    print("=" * 70)
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            stats['total'] += 1
            original_line = line.strip()
            
            if not original_line:
                continue
            
            parts = original_line.split(',')
            if len(parts) < 6:
                print(f"Line {line_num}: Invalid format, skipping")
                stats['skipped'] += 1
                continue
            
            try:
                image_path = parts[0]
                x1, y1, x2, y2 = map(int, parts[1:5])
                class_name = parts[5].strip()
                
                # Check if image exists to get dimensions
                if not os.path.exists(image_path):
                    print(f"Line {line_num}: Image not found: {image_path}, skipping")
                    stats['skipped'] += 1
                    continue
                
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Line {line_num}: Could not load image: {image_path}, skipping")
                    stats['skipped'] += 1
                    continue
                
                height, width = img.shape[:2]
                
                # Fix swapped coordinates
                if x1 > x2:
                    x1, x2 = x2, x1
                    stats['fixed'] += 1
                if y1 > y2:
                    y1, y2 = y2, y1
                    stats['fixed'] += 1
                
                # Clamp to image bounds
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width - 1))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height - 1))
                
                # Validate bounding box
                if x1 >= x2 or y1 >= y2:
                    print(f"Line {line_num}: Invalid bounding box after fixing, skipping")
                    stats['invalid'] += 1
                    continue
                
                if (x2 - x1) < 2 or (y2 - y1) < 2:
                    print(f"Line {line_num}: Bounding box too small, skipping")
                    stats['invalid'] += 1
                    continue
                
                # Write fixed line
                fixed_line = f"{image_path},{x1},{y1},{x2},{y2},{class_name}"
                fixed_lines.append(fixed_line)
                
            except (ValueError, IndexError) as e:
                print(f"Line {line_num}: Error parsing: {e}, skipping")
                stats['skipped'] += 1
                continue
    
    # Write fixed annotations
    print("\n" + "=" * 70)
    print(f"Writing {len(fixed_lines)} fixed annotations to: {output_file}")
    
    with open(output_file, 'w') as f:
        for line in fixed_lines:
            f.write(line + '\n')
    
    print("\nFix Summary:")
    print(f"  Total lines: {stats['total']}")
    print(f"  Fixed coordinates: {stats['fixed']}")
    print(f"  Invalid bounding boxes (skipped): {stats['invalid']}")
    print(f"  Other errors (skipped): {stats['skipped']}")
    print(f"  Valid annotations written: {len(fixed_lines)}")
    
    return len(fixed_lines)

def main():
    parser = argparse.ArgumentParser(
        description='Create or fix Faster R-CNN annotations from Simpsons dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create new annotations from dataset directory
  python create_annotations.py --dataset data/archive/simpsons_dataset --output annotation.txt
  
  # Fix existing annotation file
  python create_annotations.py --fix annotation.txt --output annotation_fixed.txt
  
  # Create annotations with custom train split
  python create_annotations.py --dataset data/archive/simpsons_dataset --output annotation.txt --train_split 0.9
        """
    )
    
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to Simpsons dataset directory (contains character subdirectories)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output annotation file')
    parser.add_argument('--fix', type=str, default=None,
                       help='Path to existing annotation file to fix (instead of creating new)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training split ratio (0.8 = 80%% train, 20%% test). Only affects annotation order.')
    
    args = parser.parse_args()
    
    if args.fix:
        # Fix existing annotations
        if not os.path.exists(args.fix):
            print(f"Error: Input file not found: {args.fix}")
            return 1
        
        fix_existing_annotations(args.fix, args.output)
        
    elif args.dataset:
        # Create new annotations
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset directory not found: {args.dataset}")
            return 1
        
        create_annotations_from_directory(args.dataset, args.output, args.train_split)
        
    else:
        print("Error: Must specify either --dataset (to create) or --fix (to fix existing)")
        parser.print_help()
        return 1
    
    print("\nAnnotation file created successfully!")
    return 0

if __name__ == '__main__':
    exit(main())

