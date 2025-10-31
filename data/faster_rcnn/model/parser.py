import cv2
import numpy as np
import random
import pprint
import os
import hashlib

def validate_bbox(x1, y1, x2, y2, width, height):
    """
    Validate and fix bounding box coordinates.
    Returns (valid, x1, y1, x2, y2) where valid is True if bbox is usable.
    """
    # Convert to integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Fix swapped coordinates (e.g., if x2 < x1 or y2 < y1)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    
    # Validate that bbox has valid dimensions (at least 1 pixel wide and tall)
    if x2 <= x1 or y2 <= y1:
        return False, x1, y1, x2, y2
    
    # Validate minimum size (at least 2x2 pixels)
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return False, x1, y1, x2, y2
    
    return True, x1, y1, x2, y2

def get_data(input_path):
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    skipped_lines = 0
    fixed_bboxes = 0
    invalid_bboxes = 0
    
    with open(input_path,'r') as f:
        print('Parsing annotation files')
        line_num = 0
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            line_split = line.split(',')
            
            # Validate line format
            if len(line_split) < 6:
                print('Warning: Line {} has invalid format (expected 6 comma-separated values), skipping: {}'.format(line_num, line[:50]))
                skipped_lines += 1
                continue
            
            try:
                filename = line_split[0]
                x1, y1, x2, y2 = line_split[1:5]
                class_name = line_split[5].strip()
                
                # Validate class name is not empty
                if not class_name:
                    print('Warning: Line {} has empty class name, skipping'.format(line_num))
                    skipped_lines += 1
                    continue
                
                # Check if image exists
                if not os.path.exists(filename):
                    print('Warning: Image not found: {}, skipping line {}'.format(filename, line_num))
                    skipped_lines += 1
                    continue
                
                # Load image to get dimensions (only once per image)
                if filename not in all_imgs:
                    img = cv2.imread(filename)
                    if img is None:
                        print('Warning: Could not load image: {}, skipping line {}'.format(filename, line_num))
                        skipped_lines += 1
                        continue
                    
                    (rows, cols) = img.shape[:2]
                    all_imgs[filename] = {}
                    all_imgs[filename]['filepath'] = filename
                    all_imgs[filename]['width'] = cols
                    all_imgs[filename]['height'] = rows
                    all_imgs[filename]['bboxes'] = []
                    # Use deterministic train/test split based on filename hash
                    # This ensures consistent split across runs
                    # ~83% trainval, ~17% test (matching the original ratio)
                    filename_hash = int(hashlib.md5(filename.encode()).hexdigest(), 16)
                    if (filename_hash % 6) > 0:
                        all_imgs[filename]['imageset'] = 'trainval'
                    else:
                        all_imgs[filename]['imageset'] = 'test'
                
                # Get image dimensions for validation
                width = all_imgs[filename]['width']
                height = all_imgs[filename]['height']
                
                # Validate and fix bounding box
                valid, x1_fixed, y1_fixed, x2_fixed, y2_fixed = validate_bbox(x1, y1, x2, y2, width, height)
                
                if not valid:
                    print('Warning: Line {} has invalid bounding box (too small or zero area) after fixing, skipping: {}'.format(line_num, line[:50]))
                    invalid_bboxes += 1
                    continue
                
                # Check if coordinates were fixed
                if (int(x1) != x1_fixed or int(y1) != y1_fixed or 
                    int(x2) != x2_fixed or int(y2) != y2_fixed):
                    fixed_bboxes += 1
                
                # Update class counts
                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1
                
                if class_name not in class_mapping:
                    class_mapping[class_name] = len(class_mapping)
                
                # Add validated bounding box
                all_imgs[filename]['bboxes'].append({
                    'class': class_name, 
                    'x1': x1_fixed, 
                    'x2': x2_fixed, 
                    'y1': y1_fixed, 
                    'y2': y2_fixed
                })
                
            except (ValueError, IndexError) as e:
                print('Warning: Error parsing line {}: {}, skipping'.format(line_num, str(e)))
                skipped_lines += 1
                continue
            except Exception as e:
                print('Warning: Unexpected error on line {}: {}, skipping'.format(line_num, str(e)))
                skipped_lines += 1
                continue

        all_data = []
        for key in all_imgs:
            # Skip images with no valid bounding boxes
            if len(all_imgs[key]['bboxes']) > 0:
                all_data.append(all_imgs[key])
            else:
                print('Warning: Image {} has no valid bounding boxes, skipping'.format(key))

        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
        random.shuffle(all_data)
        
        print('\nAnnotation parsing summary:')
        print('  Total valid images: {}'.format(len(all_data)))
        print('  Skipped lines: {}'.format(skipped_lines))
        print('  Fixed bounding boxes: {}'.format(fixed_bboxes))
        print('  Invalid bounding boxes: {}'.format(invalid_bboxes))
        print('\nTraining images per class ({} classes):'.format(len(classes_count)))
        pprint.pprint(classes_count)
        
        return all_data, classes_count, class_mapping
