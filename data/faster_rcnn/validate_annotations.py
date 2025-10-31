#!/usr/bin/env python3
"""
Script to validate and fix Faster R-CNN annotation file.
Checks for:
- Invalid bounding box coordinates (x2 < x1 or y2 < y1)
- Bounding boxes outside image bounds
- Missing images
- Invalid format lines
"""

import cv2
import os
import argparse
from pathlib import Path

def validate_bbox(x1, y1, x2, y2, width, height):
    """
    Validate and fix bounding box coordinates.
    Returns (valid, x1, y1, x2, y2, fixed) where:
    - valid: True if bbox is usable
    - fixed: True if coordinates were corrected
    """
    original = (int(x1), int(y1), int(x2), int(y2))
    
    # Convert to integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    fixed = False
    
    # Fix swapped coordinates (e.g., if x2 < x1 or y2 < y1)
    if x1 > x2:
        x1, x2 = x2, x1
        fixed = True
    if y1 > y2:
        y1, y2 = y2, y1
        fixed = True
    
    # Ensure coordinates are within image bounds
    x1_old, x2_old, y1_old, y2_old = x1, x2, y1, y2
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    
    if (x1 != x1_old or x2 != x2_old or y1 != y1_old or y2 != y2_old):
        fixed = True
    
    # Validate that bbox has valid dimensions (at least 2x2 pixels)
    if x2 <= x1 or y2 <= y1:
        return False, x1, y1, x2, y2, fixed
    
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return False, x1, y1, x2, y2, fixed
    
    return True, x1, y1, x2, y2, fixed

def validate_annotation_file(input_path, output_path=None, fix_errors=True):
    """
    Validate annotation file and optionally create a fixed version.
    
    Args:
        input_path: Path to input annotation file
        output_path: Path to output fixed annotation file (if None, only validates)
        fix_errors: If True, fix errors when possible; if False, only report them
    """
    if output_path is None:
        output_path = input_path + '.fixed'
    
    stats = {
        'total_lines': 0,
        'skipped_lines': 0,
        'fixed_bboxes': 0,
        'invalid_bboxes': 0,
        'missing_images': 0,
        'valid_lines': 0,
        'images_checked': set()
    }
    
    fixed_lines = []
    errors = []
    
    print('Validating annotation file: {}'.format(input_path))
    print('=' * 70)
    
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            stats['total_lines'] += 1
            original_line = line.strip()
            
            if not original_line:  # Skip empty lines
                continue
            
            line_split = original_line.split(',')
            
            # Validate line format
            if len(line_split) < 6:
                error_msg = 'Line {}: Invalid format (expected 6 comma-separated values)'.format(line_num)
                errors.append(error_msg)
                print('ERROR: {}'.format(error_msg))
                stats['skipped_lines'] += 1
                continue
            
            try:
                filename = line_split[0].strip()
                x1_str, y1_str, x2_str, y2_str = line_split[1:5]
                class_name = line_split[5].strip()
                
                # Validate class name is not empty
                if not class_name:
                    error_msg = 'Line {}: Empty class name'.format(line_num)
                    errors.append(error_msg)
                    print('ERROR: {}'.format(error_msg))
                    stats['skipped_lines'] += 1
                    continue
                
                # Check if image exists
                if not os.path.exists(filename):
                    error_msg = 'Line {}: Image not found: {}'.format(line_num, filename)
                    errors.append(error_msg)
                    print('ERROR: {}'.format(error_msg))
                    stats['missing_images'] += 1
                    stats['skipped_lines'] += 1
                    continue
                
                # Load image to get dimensions (cache by filename)
                if filename not in stats['images_checked']:
                    img = cv2.imread(filename)
                    if img is None:
                        error_msg = 'Line {}: Could not load image: {}'.format(line_num, filename)
                        errors.append(error_msg)
                        print('ERROR: {}'.format(error_msg))
                        stats['skipped_lines'] += 1
                        continue
                    stats['images_checked'].add(filename)
                    height, width = img.shape[:2]
                else:
                    # Re-read image dimensions (could cache this if needed)
                    img = cv2.imread(filename)
                    if img is None:
                        error_msg = 'Line {}: Could not load image: {}'.format(line_num, filename)
                        errors.append(error_msg)
                        stats['skipped_lines'] += 1
                        continue
                    height, width = img.shape[:2]
                
                # Validate and fix bounding box
                try:
                    x1, y1, x2, y2 = int(x1_str), int(y1_str), int(x2_str), int(y2_str)
                except ValueError:
                    error_msg = 'Line {}: Invalid coordinate values (not integers)'.format(line_num)
                    errors.append(error_msg)
                    print('ERROR: {}'.format(error_msg))
                    stats['skipped_lines'] += 1
                    continue
                
                valid, x1_fixed, y1_fixed, x2_fixed, y2_fixed, was_fixed = validate_bbox(
                    x1, y1, x2, y2, width, height
                )
                
                if not valid:
                    error_msg = 'Line {}: Invalid bounding box (too small or zero area): {}'.format(
                        line_num, original_line[:60]
                    )
                    errors.append(error_msg)
                    print('ERROR: {}'.format(error_msg))
                    stats['invalid_bboxes'] += 1
                    stats['skipped_lines'] += 1
                    continue
                
                if was_fixed:
                    stats['fixed_bboxes'] += 1
                    print('FIXED: Line {} - Original: ({},{},{},{}) -> Fixed: ({},{},{},{})'.format(
                        line_num, x1, y1, x2, y2, x1_fixed, y1_fixed, x2_fixed, y2_fixed
                    ))
                
                # Write fixed line
                if fix_errors:
                    fixed_line = '{},{},{},{},{},{}'.format(
                        filename, x1_fixed, y1_fixed, x2_fixed, y2_fixed, class_name
                    )
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(original_line)
                
                stats['valid_lines'] += 1
                
            except (ValueError, IndexError) as e:
                error_msg = 'Line {}: Parsing error: {}'.format(line_num, str(e))
                errors.append(error_msg)
                print('ERROR: {}'.format(error_msg))
                stats['skipped_lines'] += 1
                continue
            except Exception as e:
                error_msg = 'Line {}: Unexpected error: {}'.format(line_num, str(e))
                errors.append(error_msg)
                print('ERROR: {}'.format(error_msg))
                stats['skipped_lines'] += 1
                continue
    
    # Write fixed annotation file
    if fix_errors and fixed_lines:
        print('\n' + '=' * 70)
        print('Writing fixed annotation file: {}'.format(output_path))
        with open(output_path, 'w') as f:
            for line in fixed_lines:
                f.write(line + '\n')
        print('Fixed annotation file saved successfully!')
    
    # Print summary
    print('\n' + '=' * 70)
    print('VALIDATION SUMMARY')
    print('=' * 70)
    print('Total lines processed: {}'.format(stats['total_lines']))
    print('Valid lines: {}'.format(stats['valid_lines']))
    print('Skipped lines: {}'.format(stats['skipped_lines']))
    print('  - Missing images: {}'.format(stats['missing_images']))
    print('  - Invalid bounding boxes: {}'.format(stats['invalid_bboxes']))
    print('Fixed bounding boxes: {}'.format(stats['fixed_bboxes']))
    print('Unique images checked: {}'.format(len(stats['images_checked'])))
    
    if errors and not fix_errors:
        print('\nTotal errors found: {}'.format(len(errors)))
    
    return stats, errors

def main():
    parser = argparse.ArgumentParser(
        description='Validate and fix Faster R-CNN annotation file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate only (report errors)
  python validate_annotations.py annotation.txt --no-fix
  
  # Validate and create fixed file
  python validate_annotations.py annotation.txt -o annotation_fixed.txt
        """
    )
    parser.add_argument('input', help='Path to input annotation file')
    parser.add_argument('-o', '--output', default=None,
                       help='Path to output fixed annotation file (default: input.fixed)')
    parser.add_argument('--no-fix', action='store_true',
                       help='Only validate, do not fix errors')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print('Error: Input file not found: {}'.format(args.input))
        return 1
    
    fix_errors = not args.no_fix
    stats, errors = validate_annotation_file(args.input, args.output, fix_errors=fix_errors)
    
    if stats['valid_lines'] == 0:
        print('\nWARNING: No valid annotations found!')
        return 1
    
    if stats['skipped_lines'] > 0:
        print('\nWARNING: Some annotations were skipped. Review the errors above.')
        return 1 if not fix_errors else 0
    
    print('\nValidation completed successfully!')
    return 0

if __name__ == '__main__':
    exit(main())

