#!/usr/bin/env python3
"""
Main script to run both Faster R-CNN and YOLOv3 models
for object detection tasks.

Usage:
    python main.py --model faster_rcnn --mode train --config path/to/config
    python main.py --model yolov3 --mode detect --image path/to/image.jpg
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# Add data directories to Python path
sys.path.append(str(Path(__file__).parent / "data" / "faster_rcnn"))
sys.path.append(str(Path(__file__).parent / "data" / "yolov3"))

warnings.filterwarnings("ignore")

def run_faster_rcnn(args):
    """Run Faster R-CNN model"""
    print("="*50)
    print("Running Faster R-CNN")
    print("="*50)
    
    os.chdir("data/faster_rcnn")
    
    if args.mode == "train":
        if not args.config:
            print("Error: --config required for training mode")
            return
        
        cmd = f"python train.py -p {args.config}"
        if args.epochs:
            cmd += f" --n_epochs {args.epochs}"
        if args.iters:
            cmd += f" --n_iters {args.iters}"
            
        print(f"Training command: {cmd}")
        os.system(cmd)
        
    elif args.mode == "detect":
        if not args.config or not args.input_dir:
            print("Error: --config and --input_dir required for detection mode")
            return
            
        cmd = f"python predict.py -c {args.config} -i {args.input_dir}"
        print(f"Detection command: {cmd}")
        os.system(cmd)
    
    os.chdir("../..")

def run_yolov3(args):
    """Run YOLOv3 model"""
    print("="*50)
    print("Running YOLOv3")
    print("="*50)
    
    os.chdir("data/yolov3")
    
    if args.mode == "train":
        print("YOLOv3 training requires proper dataset preparation")
        print("Please refer to the README for dataset format requirements")
        cmd = "python train.py"
        print(f"Training command: {cmd}")
        os.system(cmd)
        
    elif args.mode == "detect":
        if args.image:
            cmd = f"python yolo_video.py --image --input {args.image}"
        elif args.video:
            cmd = f"python yolo_video.py {args.video}"
            if args.output:
                cmd += f" {args.output}"
        else:
            # Use default or test image
            cmd = "python yolo_video.py --image"
            
        print(f"Detection command: {cmd}")
        os.system(cmd)
    
    os.chdir("../..")

def setup_environment():
    """Setup environment and check requirements"""
    print("Setting up environment...")
    
    # Create necessary directories
    os.makedirs("plots", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/test_images", exist_ok=True)
    
    # Check if models are available
    faster_rcnn_path = Path("data/faster_rcnn")
    yolov3_path = Path("data/yolov3")
    
    if not faster_rcnn_path.exists():
        print("Error: Faster R-CNN implementation not found")
        return False
        
    if not yolov3_path.exists():
        print("Error: YOLOv3 implementation not found")
        return False
        
    print("Environment setup complete!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Object Detection with Faster R-CNN and YOLOv3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Faster R-CNN
  python main.py --model faster_rcnn --mode train --config annotation.txt

  # Detect with Faster R-CNN  
  python main.py --model faster_rcnn --mode detect --config config.pickle --input_dir images/

  # Detect with YOLOv3 on image
  python main.py --model yolov3 --mode detect --image test.jpg

  # Detect with YOLOv3 on video
  python main.py --model yolov3 --mode detect --video input.mp4 --output output.mp4
        """
    )
    
    parser.add_argument("--model", choices=["faster_rcnn", "yolov3"], 
                       required=True, help="Model to use")
    parser.add_argument("--mode", choices=["train", "detect"], 
                       required=True, help="Mode to run")
    
    # Faster R-CNN specific arguments
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--iters", type=int, help="Number of training iterations")
    parser.add_argument("--input_dir", help="Input directory for detection")
    
    # YOLOv3 specific arguments
    parser.add_argument("--image", help="Input image for detection")
    parser.add_argument("--video", help="Input video for detection")
    parser.add_argument("--output", help="Output video path")
    
    args = parser.parse_args()
    
    if not setup_environment():
        return
    
    start_time = time.time()
    
    try:
        if args.model == "faster_rcnn":
            run_faster_rcnn(args)
        elif args.model == "yolov3":
            run_yolov3(args)
            
        elapsed_time = time.time() - start_time
        print(f"\nExecution completed in {elapsed_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
