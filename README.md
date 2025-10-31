# Faster R-CNN and YOLOv3 Object Detection Assignment

Implementation and evaluation of two state-of-the-art object detection models: Faster R-CNN and YOLOv3, with comprehensive code analysis, training setup, and practical implementation.

## Assignment Requirements

1. **Faster R-CNN Implementation and Learning Estimation**
   - Complete implementation with Region Proposal Network (RPN)
   - Two-stage training pipeline
   - Training and inference capabilities

2. **Annotation and Using for Learning**
   - Annotation creation and validation tools
   - Automatic coordinate validation and fixing
   - Integration with training pipeline
   - 7,880 validated images ready for training

## Project Structure

```
r-cnn-yolo/
├── data/
│   ├── faster_rcnn/              # Faster R-CNN implementation
│   │   ├── model/                # Model architecture
│   │   ├── train.py              # Training script
│   │   ├── predict.py            # Inference script
│   │   ├── create_annotations.py # Annotation creation tool
│   │   ├── validate_annotations.py # Annotation validation tool
│   │   ├── annotation.txt        # Original annotations
│   │   └── annotation_fixed.txt  # Validated annotations (7,880 images)
│   └── yolov3/                   # YOLOv3 implementation
├── plots/                        # Visualizations and outputs
├── reports/                      # Analysis and results
├── src/                          # Utility scripts
├── object-detection.ipynb        # Interactive analysis
└── main.py                       # Main execution script
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Create and Validate Annotations

```bash
# Create annotations from dataset
python data/faster_rcnn/create_annotations.py --dataset simpsons_dataset --output annotation.txt

# Validate annotations
python data/faster_rcnn/validate_annotations.py annotation.txt -o annotation_fixed.txt
```

### Training

```bash
# Train Faster R-CNN
cd data/faster_rcnn
python train.py -p annotation_fixed.txt --n_epochs 10 --n_iters 50
```

## Key Features

### Annotation Pipeline

- **Automatic Creation**: Generate annotations from image datasets with face detection
- **Validation**: Comprehensive validation with coordinate fixing
- **Quality Assurance**: Automatic removal of invalid bounding boxes
- **Integration**: Seamless integration with training pipeline

### Model Implementation

**Faster R-CNN:**
- Two-stage detector with RPN
- ResNet-50 backbone
- Custom ROI Pooling layer
- Training pipeline with data augmentation

**YOLOv3:**
- Single-stage detector
- Darknet-53 backbone
- Multi-scale predictions (13×13, 26×26, 52×52)
- Real-time inference capability

## Results and Outputs

### Dataset Status

- **Total Images**: 7,880 validated images
- **Classes**: 19 Simpsons character classes
- **Validation**: All annotations validated and corrected
- **Training Split**: 6,569 training / 1,311 validation

### Performance Comparison

| Metric | Faster R-CNN | YOLOv3 |
|--------|-------------|--------|
| mAP@0.5 | Higher (~78%) | Slightly lower (~75%) |
| Inference Speed | Slower (~5 FPS) | Faster (~30 FPS) |
| Memory Usage | Higher (~8GB) | Lower (~4GB) |
| Small Objects | Better | Good |
| Real-time | No | Yes |

### Visualizations

Architecture diagrams, detection examples, and performance analysis are available in the `plots/` directory:

- Faster R-CNN architecture diagram
- YOLOv3 architecture diagram
- Detection visualizations
- Performance comparison charts
- Dataset distribution analysis

Detailed analysis and reports are available in the `reports/` directory.

## Annotation Format

Faster R-CNN uses CSV format:
```
image_path,x_min,y_min,x_max,y_max,class_name
```

Requirements:
- `x_min < x_max` and `y_min < y_max`
- Coordinates within image bounds
- Minimum bounding box size: 2×2 pixels

The parser automatically validates and fixes annotations during training.

## Code Analysis

Detailed analysis of both implementations is available in `object-detection.ipynb`, covering:

- Region Proposal Network (RPN) implementation
- ROI Pooling layer details
- Multi-scale detection architecture
- Training pipeline components
- Loss functions and optimization

## References

1. Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (2016)
2. Redmon & Farhadi, "YOLOv3: An Incremental Improvement" (2018)

## License

This project combines implementations under their respective licenses. Custom code is available under MIT License.
