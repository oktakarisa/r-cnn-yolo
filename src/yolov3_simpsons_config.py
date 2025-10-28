#!/usr/bin/env python3
"""
YOLOv3 configuration for Simpsons dataset training
"""

import os
import sys
from pathlib import Path

# Add yolov3 path
sys.path.append(str(Path(__file__).parent.parent / "data" / "yolov3"))

class SimpsonsConfig:
    """Configuration for YOLOv3 Simpsons dataset training"""
    
    # Dataset paths
    DATASET_DIR = "data/yolov3/simpsons"
    CLASSES_FILE = f"{DATASET_DIR}/classes.txt"
    TRAIN_FILE = f"{DATASET_DIR}/train.txt"
    VAL_FILE = f"{DATASET_DIR}/val.txt"
    
    # Model paths
    ANCHORS_PATH = "data/yolov3/model_data/yolo_anchors.txt"
    MODEL_PATH = "data/yolov3/model_data/yolo_weights.h5"
    CONVERTED_MODEL = "model_data/yolo_weights.h5"
    
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    INPUT_SHAPE = (416, 416)  # height, width
    
    # YOLO specific parameters
    SCORE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.45
    MAX_BOXES = 100
    
    # Hardware
    GPU_NUM = 1
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        print("Validating Simpsons dataset configuration...")
        
        checks = [
            (cls.CLASSES_FILE, "Classes file"),
            (cls.TRAIN_FILE, "Training file"),
            (cls.VAL_FILE, "Validation file"),
            (cls.ANCHORS_PATH, "Anchors file"),
        ]
        
        all_valid = True
        for path, name in checks:
            if os.path.exists(path):
                print(f"✓ {name}: {path}")
            else:
                print(f"✗ {name}: {path} (not found)")
                all_valid = False
        
        return all_valid
    
    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        print("\n" + "="*50)
        print("YOLOv3 Simpsons Configuration")
        print("="*50)
        print(f"Dataset directory: {cls.DATASET_DIR}")
        print(f"Classes file: {cls.CLASSES_FILE}")
        print(f"Training file: {cls.TRAIN_FILE}")
        print(f"Validation file: {cls.VAL_FILE}")
        print(f"Model path: {cls.MODEL_PATH}")
        print(f"Input shape: {cls.INPUT_SHAPE}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Score threshold: {cls.SCORE_THRESHOLD}")
        print(f"IOU threshold: {cls.IOU_THRESHOLD}")
        print("="*50)

def create_modified_train_script():
    """Create a modified train.py for Simpsons dataset"""
    
    script_content = '''#!/usr/bin/env python3
"""
Modified YOLOv3 training script for Simpsons dataset
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body
from yolo3.utils import get_random_data

def _main():
    from src.yolov3_simpsons_config import SimpsonsConfig as C
    import os
    
    # Validate configuration
    if not C.validate():
        print("Configuration validation failed!")
        return
    
    C.print_config()
    
    annotation_path = C.TRAIN_FILE
    log_dir = 'logs/000'
    classes_path = C.CLASSES_FILE
    anchors_path = C.ANCHORS_PATH
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = C.INPUT_SHAPE + (3,) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=C.MODEL_PATH)
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=C.MODEL_PATH) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    if True:
        model.compile(optimizer=Adam(lr=C.LEARNING_RATE), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = C.BATCH_SIZE
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=C.EPOCHS,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = C.BATCH_SIZE // 4  # reduce batch size to save memory
        print('Re-train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=C.EPOCHS,
                initial_epoch=50,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape[:2]
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape[:2]
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            num = (80, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    for i in range(0, n, batch_size):
        image_data = []
        box_data = []
        for j in range(i, min(i + batch_size, n)):
            image, box = get_random_data(annotation_lines[j], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

if __name__ == '__main__':
    _main()
'''
    
    with open("data/yolov3/train_simpsons.py", "w") as f:
        f.write(script_content)
    
    print("Created Simpsons training script: data/yolov3/train_simpsons.py")

if __name__ == "__main__":
    create_modified_train_script()
    SimpsonsConfig.print_config()
