#!/usr/bin/env python3
"""
Test script to verify that the training setup works correctly
"""

import os
import sys
import traceback
from pathlib import Path

# Add yolov3 path
sys.path.append(str(Path(__file__).parent.parent / "data" / "yolov3"))

def test_yolov3_setup():
    """Test YOLOv3 training setup"""
    print("Testing YOLOv3 setup...")
    
    try:
        # Test if we can import YOLOv3 modules
        from yolo3.model import yolo_body
        from yolo3.utils import get_random_data
        print("[PASS] YOLOv3 modules imported successfully")
        
        # Test if model files exist
        if os.path.exists("data/yolov3/model_data/yolo_anchors.txt"):
            print("[PASS] Anchor file found")
        else:
            print("[FAIL] Anchor file not found")
            
        if os.path.exists("data/yolov3/yolov3.cfg"):
            print("[PASS] YOLOv3 config found")
        else:
            print("[FAIL] YOLOv3 config not found")
            
        return True
        
    except Exception as e:
        print(f"[FAIL] YOLOv3 setup test failed: {e}")
        traceback.print_exc()
        return False

def test_faster_rcnn_setup():
    """Test Faster R-CNN training setup"""
    print("\nTesting Faster R-CNN setup...")
    
    try:
        # Test if we can import Faster R-CNN modules
        sys.path.append("data/faster_rcnn")
        from model import faster_rcnn
        from model import config
        print("[PASS] Faster R-CNN modules imported successfully")
        
        # Test if annotation file exists
        if os.path.exists("data/faster_rcnn/annotation.txt"):
            print("[PASS] Annotation file found")
            # Count annotations
            with open("data/faster_rcnn/annotation.txt", 'r') as f:
                count = len([line for line in f if line.strip()])
            print(f"  Found {count} annotations")
        else:
            print("[FAIL] Annotation file not found")
            
        return True
        
    except Exception as e:
        print(f"[FAIL] Faster R-CNN setup test failed: {e}")
        traceback.print_exc()
        return False

def test_simpsons_dataset_conversion():
    """Test Simpsons dataset conversion setup"""
    print("\nTesting Simpsons dataset conversion...")
    
    try:
        # Test conversion script
        sys.path.append(str(Path(__file__).parent))
        from prepare_simpsons_dataset import read_faster_rcnn_annotations
        
        # Test reading annotations
        if os.path.exists("data/faster_rcnn/annotation.txt"):
            classes, annotations = read_faster_rcnn_annotations("data/faster_rcnn/annotation.txt")
            print(f"[PASS] Found {len(classes)} classes: {', '.join(classes[:5])}...")
            print(f"  Found {len(annotations)} annotations")
        else:
            print("[FAIL] Cannot test conversion - annotation file not found")
            
        return True
        
    except Exception as e:
        print(f"[FAIL] Dataset conversion test failed: {e}")
        traceback.print_exc()
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("\nCreating sample data for testing...")
    
    # Create a simple test image
    try:
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_img = Image.new('RGB', (224, 224), color='red')
        test_img.save('data/test_images/test_image.jpg')
        print("[PASS] Created test image")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Failed to create sample data: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("Training Setup Verification")
    print("="*50)
    
    # Create necessary directories
    os.makedirs("data/test_images", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    results = []
    
    # Run tests
    results.append(("YOLOv3 Setup", test_yolov3_setup()))
    results.append(("Faster R-CNN Setup", test_faster_rcnn_setup()))
    results.append(("Dataset Conversion", test_simpsons_dataset_conversion()))
    results.append(("Sample Data", create_sample_data()))
    
    # Print summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("[PASS] All tests passed! Training setup is ready.")
        return True
    else:
        print("[FAIL] Some tests failed. Please check the setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
