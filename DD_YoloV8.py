"""
===========================================================
Water Sensitive Paper (WSP) Droplet Detection using YOLOv8
Author  : Muhammad Adli
Framework : Ultralytics YOLOv8.4.50
GPU : RTX 4060 Laptop (8 GB)
Python : 3.10+
===========================================================
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# ==========================================================
# CONFIGURATION
# ==========================================================

# -------------------------------
# Dataset
# -------------------------------

DATASET_YAML = "dataset/dataset.yaml"

TRAIN_FOLDER = "dataset/images/train"
VAL_FOLDER = "dataset/images/val"
TEST_FOLDER = "dataset/images/test"

# default image after training
SAMPLE_IMAGE = os.path.join(TEST_FOLDER, "270.jpg")

# -------------------------------
# Model
# -------------------------------

# MODEL_NAME = "yolov8s-seg.pt"

# Alternative
MODEL_NAME = "yolov8n-seg.pt"
# MODEL_NAME = "yolov8m-seg.pt"

# -------------------------------
# Training Parameters
# -------------------------------

IMAGE_SIZE = 640
BATCH_SIZE = 2

EPOCHS = 50
PATIENCE = 20

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4

OPTIMIZER = "AdamW"

# -------------------------------
# Augmentation
# -------------------------------

MOSAIC = 0.2
CLOSE_MOSAIC = 10

# -------------------------------
# Prediction
# -------------------------------

CONFIDENCE = 0.25
IOU = 0.50

RETINA_MASKS = True

# -------------------------------
# Output
# -------------------------------

PROJECT_NAME = "DropletDetection_YOLOv8"

RUN_NAME = f"{MODEL_NAME.replace('.pt','')}_{IMAGE_SIZE}px"

CSV_OUTPUT = "prediction_summary.csv"

# ==========================================================
# GPU SETUP
# ==========================================================

def setup_gpu():

    print("=" * 60)

    if not torch.cuda.is_available():

        print("CUDA NOT AVAILABLE")
        print("Training will run on CPU.")

        return "cpu"

    device = torch.device("cuda:0")

    print("CUDA AVAILABLE")

    print("GPU :", torch.cuda.get_device_name(0))
    print("CUDA Version :", torch.version.cuda)
    print("GPU Count :", torch.cuda.device_count())

    total_memory = torch.cuda.get_device_properties(
        0
    ).total_memory / (1024 ** 3)

    print(f"GPU Memory : {total_memory:.2f} GB")

    # reserve around 85%
    torch.cuda.set_per_process_memory_fraction(
        0.85,
        device=0
    )

    torch.cuda.empty_cache()

    print("=" * 60)

    return device

# ==========================================================
# VERIFY DATASET
# ==========================================================

def verify_dataset():

    print("\nChecking Dataset...")

    if not os.path.exists(DATASET_YAML):

        raise FileNotFoundError(
            f"{DATASET_YAML} not found!"
        )

    for folder in [
        TRAIN_FOLDER,
        VAL_FOLDER,
        TEST_FOLDER
    ]:

        if not os.path.exists(folder):

            raise FileNotFoundError(
                f"{folder} not found!"
            )

    train_images = glob.glob(
        os.path.join(TRAIN_FOLDER, "*.jpg")
    )

    val_images = glob.glob(
        os.path.join(VAL_FOLDER, "*.jpg")
    )

    test_images = glob.glob(
        os.path.join(TEST_FOLDER, "*.jpg")
    )

    print(f"Training Images : {len(train_images)}")
    print(f"Validation Images : {len(val_images)}")
    print(f"Testing Images : {len(test_images)}")

    print("Dataset OK\n")

# ==========================================================
# OPTIONAL PREPROCESSING
# ==========================================================

def preprocess_wsp(image_path):

    image = cv2.imread(image_path)

    if image is None:

        raise FileNotFoundError(image_path)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_blue = np.array([90,50,50])
    upper_blue = np.array([140,255,255])

    mask = cv2.inRange(
        hsv,
        lower_blue,
        upper_blue
    )

    mask = cv2.medianBlur(mask,3)

    kernel = np.ones(
        (3,3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return image, mask

# ==========================================================
# CREATE MODEL
# ==========================================================

def build_model():

    print("Loading Model...")

    model = YOLO(MODEL_NAME)

    print(model.model)

    return model

# ==========================================================
# FIND BEST MODEL
# ==========================================================

def find_best_model():

    weights = glob.glob(
        os.path.join(
            PROJECT_NAME,
            RUN_NAME,
            "weights",
            "best.pt"
        )
    )

    if len(weights) == 0:

        raise FileNotFoundError(
            "best.pt not found!"
        )

    return weights[0]

# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(model):
    """
    Train YOLOv8 Segmentation Model
    """

    print("=" * 60)
    print("Starting Training...")
    print("=" * 60)

    results = model.train(

        # -------------------------
        # Dataset
        # -------------------------
        data=DATASET_YAML,

        # -------------------------
        # Training
        # -------------------------
        epochs=EPOCHS,
        patience=PATIENCE,

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        optimizer=OPTIMIZER,
        lr0=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        # -------------------------
        # Segmentation
        # -------------------------
        overlap_mask=True,
        mask_ratio=4,

        # -------------------------
        # Augmentation
        # -------------------------
        mosaic=MOSAIC,
        close_mosaic=CLOSE_MOSAIC,

        degrees=0.0,
        translate=0.10,
        scale=0.30,
        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.50,

        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,

        copy_paste=0.0,

        # -------------------------
        # Performance
        # -------------------------
        amp=True,
        cache=False,
        workers=0,

        # -------------------------
        # Save
        # -------------------------
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,

        save=True,
        save_period=10,

        plots=True,
        verbose=True
    )

    print("\nTraining Finished.")

    return results


# ==========================================================
# VALIDATE MODEL
# ==========================================================

def validate_model(model):
    """
    Validate trained model
    """

    print("=" * 60)
    print("Validation")
    print("=" * 60)

    metrics = model.val(
        data=DATASET_YAML,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        split="val",
        save_json=True,
        plots=True
    )

    print("\nValidation Completed")

    return metrics


# ==========================================================
# TEST MODEL
# ==========================================================

def test_model(model):
    """
    Evaluate on test dataset
    """

    print("=" * 60)
    print("Testing")
    print("=" * 60)

    metrics = model.val(

        data=DATASET_YAML,

        split="test",

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        save_json=True,

        plots=True
    )

    print("\nTesting Completed")

    return metrics


# ==========================================================
# PRINT METRICS
# ==========================================================

def print_metrics(metrics):
    """
    Print evaluation metrics
    """

    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)

    try:

        print(f"mAP50        : {metrics.box.map50:.4f}")
        print(f"mAP50-95     : {metrics.box.map:.4f}")

    except Exception:
        pass

    try:

        print(f"Seg mAP50    : {metrics.seg.map50:.4f}")
        print(f"Seg mAP50-95 : {metrics.seg.map:.4f}")

    except Exception:
        pass

    try:

        print(f"Precision    : {metrics.box.mp:.4f}")
        print(f"Recall       : {metrics.box.mr:.4f}")

    except Exception:
        pass

    print("=" * 60)


# ==========================================================
# LOAD BEST MODEL
# ==========================================================

def load_best_model():
    """
    Automatically load best.pt
    """

    best_path = find_best_model()

    print("\nLoading Best Model")
    print(best_path)

    model = YOLO(best_path)

    return model


# ==========================================================
# SAVE TRAINING CONFIGURATION
# ==========================================================

def save_configuration():
    """
    Save training parameters
    """

    config = {

        "Model": MODEL_NAME,
        "Image Size": IMAGE_SIZE,
        "Batch Size": BATCH_SIZE,
        "Epochs": EPOCHS,
        "Optimizer": OPTIMIZER,
        "Learning Rate": LEARNING_RATE,
        "Weight Decay": WEIGHT_DECAY,
        "Mosaic": MOSAIC,
        "Close Mosaic": CLOSE_MOSAIC,
        "Confidence": CONFIDENCE,
        "IOU": IOU,
        "Retina Masks": RETINA_MASKS,
        "Date": datetime.now()
    }

    df = pd.DataFrame(
        config.items(),
        columns=["Parameter", "Value"]
    )

    output = os.path.join(
        PROJECT_NAME,
        RUN_NAME,
        "training_configuration.csv"
    )

    os.makedirs(
        os.path.dirname(output),
        exist_ok=True
    )

    df.to_csv(
        output,
        index=False
    )

    print("\nTraining configuration saved.")


# ==========================================================
# TRAINING PIPELINE
# ==========================================================

def run_training():

    verify_dataset()

    device = setup_gpu()

    model = build_model()

    save_configuration()

    train_model(model)

    metrics = validate_model(model)

    print_metrics(metrics)

    try:

        test_metrics = test_model(model)

        print_metrics(test_metrics)

    except Exception as e:

        print("Test dataset not found.")
        print(e)

    print("\nTraining Pipeline Finished.")

    return load_best_model()

# ==========================================================
# INFERENCE FUNCTIONS
# ==========================================================


def predict_image(model, image_path):
    """
    Run YOLO segmentation prediction
    """

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    print("=" * 60)
    print("Running Prediction")
    print("=" * 60)

    results = model.predict(

        source=image_path,

        imgsz=IMAGE_SIZE,

        conf=CONFIDENCE,

        iou=IOU,

        retina_masks=RETINA_MASKS,

        save=True,

        save_txt=True,

        save_conf=True,

        verbose=False
    )


    print("Prediction completed.")

    return results



# ==========================================================
# MASK PROCESSING
# ==========================================================


def extract_mask(results):
    """
    Combine all predicted masks
    """

    if results[0].masks is None:

        print("No droplets detected.")

        return np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8
        )


    masks = (
        results[0]
        .masks
        .data
        .cpu()
        .numpy()
    )


    combined_mask = np.any(
        masks,
        axis=0
    )


    binary_mask = (
        combined_mask
        .astype(np.uint8)
        *
        255
    )


    return binary_mask



# ==========================================================
# DROPLET COUNTING
# ==========================================================


def count_droplets(results):
    """
    Count detected droplet instances
    """

    if results[0].masks is None:

        return 0


    count = len(
        results[0]
        .masks
        .data
    )


    return count



# ==========================================================
# COVERAGE CALCULATION
# ==========================================================


def calculate_coverage(binary_mask):
    """
    Calculate spray coverage percentage

    Formula:

    Coverage =
    Droplet pixels /
    Total pixels x 100
    """


    droplet_pixels = np.sum(
        binary_mask == 255
    )


    total_pixels = binary_mask.size


    coverage = (
        droplet_pixels /
        total_pixels
    ) * 100


    return coverage



# ==========================================================
# VISUALIZATION
# ==========================================================


def visualize_prediction(
        image_path,
        results,
        binary_mask
):

    """
    Show:
    Original image
    YOLO prediction
    Binary mask
    """


    original = cv2.imread(
        image_path
    )


    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )


    prediction = results[0].plot()


    prediction_rgb = cv2.cvtColor(
        prediction,
        cv2.COLOR_BGR2RGB
    )


    mask_rgb = cv2.cvtColor(
        binary_mask,
        cv2.COLOR_GRAY2RGB
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18,6)
    )


    axes[0].imshow(
        original_rgb
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis(
        "off"
    )


    axes[1].imshow(
        prediction_rgb
    )

    axes[1].set_title(
        "YOLOv8 Segmentation"
    )

    axes[1].axis(
        "off"
    )


    axes[2].imshow(
        mask_rgb
    )

    axes[2].set_title(
        "Binary Mask"
    )

    axes[2].axis(
        "off"
    )


    plt.tight_layout()

    plt.show()



# ==========================================================
# SAVE RESULT SUMMARY
# ==========================================================


def save_prediction_result(
        image_path,
        droplet_count,
        coverage
):

    """
    Save prediction statistics
    """

    result = {

        "Image":
            os.path.basename(
                image_path
            ),

        "Droplet Count":
            droplet_count,

        "Coverage (%)":
            round(
                coverage,
                4
            ),

        "Date":
            datetime.now()

    }


    df = pd.DataFrame(
        [result]
    )


    if os.path.exists(
        CSV_OUTPUT
    ):

        old = pd.read_csv(
            CSV_OUTPUT
        )

        df = pd.concat(
            [
                old,
                df
            ],
            ignore_index=True
        )


    df.to_csv(
        CSV_OUTPUT,
        index=False
    )


    print(
        f"Result saved: {CSV_OUTPUT}"
    )



# ==========================================================
# COMPLETE INFERENCE PIPELINE
# ==========================================================


def run_inference(model, image_path):


    results = predict_image(
        model,
        image_path
    )


    mask = extract_mask(
        results
    )


    droplets = count_droplets(
        results
    )


    coverage = calculate_coverage(
        mask
    )


    print("\n" + "="*60)

    print(
        f"Droplet Count : {droplets}"
    )

    print(
        f"Coverage      : {coverage:.3f}%"
    )

    print("="*60)



    visualize_prediction(
        image_path,
        results,
        mask
    )


    save_prediction_result(
        image_path,
        droplets,
        coverage
    )


    return {

        "droplets":
            droplets,

        "coverage":
            coverage
    }

# ==========================================================
# MAIN PROGRAM
# ==========================================================

import argparse



def main():

    parser = argparse.ArgumentParser(
        description=
        "YOLOv8 Water Sensitive Paper Droplet Segmentation"
    )


    parser.add_argument(
        "--train",
        action="store_true",
        help="Train YOLOv8 segmentation model"
    )


    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help=
        "Predict single image"
    )


    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=
        "Evaluate trained model"
    )


    args = parser.parse_args()



    # ------------------------------------------------------
    # TRAINING MODE
    # ------------------------------------------------------

    if args.train:

        print("\n")
        print("="*60)
        print("TRAINING MODE")
        print("="*60)


        model = run_training()


        print("\nRunning inference example...")


        if os.path.exists(
            SAMPLE_IMAGE
        ):

            run_inference(
                model,
                SAMPLE_IMAGE
            )


        else:

            print(
                "Sample image not found:"
            )

            print(
                SAMPLE_IMAGE
            )



    # ------------------------------------------------------
    # PREDICTION MODE
    # ------------------------------------------------------

    elif args.predict:


        print("\n")
        print("="*60)
        print("PREDICTION MODE")
        print("="*60)


        best_model_path = (
            find_best_model()
        )


        model = YOLO(
            best_model_path
        )


        run_inference(
            model,
            args.predict
        )



    # ------------------------------------------------------
    # EVALUATION MODE
    # ------------------------------------------------------

    elif args.evaluate:


        print("\n")
        print("="*60)
        print("EVALUATION MODE")
        print("="*60)


        model_path = (
            find_best_model()
        )


        model = YOLO(
            model_path
        )


        metrics = validate_model(
            model
        )


        print_metrics(
            metrics
        )


        test_metrics = test_model(
            model
        )


        print_metrics(
            test_metrics
        )



    else:

        print("\nNo command selected.")

        print("\nExamples:")

        print(
            "Training:"
        )

        print(
            "python wsp_yolov8_seg.py --train"
        )


        print(
            "\nPrediction:"
        )

        print(
            "python wsp_yolov8_seg.py --predict dataset/images/test/270.jpg"
        )


        print(
            "\nEvaluation:"
        )

        print(
            "python wsp_yolov8_seg.py --evaluate"
        )



# ==========================================================
# PROGRAM ENTRY POINT
# ==========================================================


if __name__ == "__main__":

    main()

# ==========================================================
# PART 5
# DATASET ANALYSIS AND QUALITY CHECK
# ==========================================================


def check_label_files():

    """
    Check YOLO segmentation label files
    """

    print("=" * 60)
    print("Checking Label Files")
    print("=" * 60)


    image_extensions = [
        "*.jpg",
        "*.png",
        "*.jpeg"
    ]


    images = []


    for ext in image_extensions:

        images.extend(
            glob.glob(
                os.path.join(
                    "dataset/images",
                    "**",
                    ext
                ),
                recursive=True
            )
        )


    missing_labels = []
    empty_labels = []


    total_objects = 0


    for image in images:


        image_name = Path(
            image
        ).stem


        label = os.path.join(
            "dataset/labels",
            Path(image).parent.name,
            image_name + ".txt"
        )


        if not os.path.exists(label):

            missing_labels.append(
                image
            )

            continue



        with open(
            label,
            "r"
        ) as f:

            lines = f.readlines()



        if len(lines) == 0:

            empty_labels.append(
                image
            )

        else:

            total_objects += len(lines)



    print(
        f"Total Images     : {len(images)}"
    )

    print(
        f"Total Annotations: {total_objects}"
    )


    print(
        f"Missing Labels   : {len(missing_labels)}"
    )


    print(
        f"Empty Labels     : {len(empty_labels)}"
    )


    if len(missing_labels) > 0:

        print("\nMissing label example:")

        print(
            missing_labels[:5]
        )


    if len(empty_labels) > 0:

        print("\nEmpty label example:")

        print(
            empty_labels[:5]
        )


    print("=" * 60)



# ==========================================================
# ANALYZE DROPLET DISTRIBUTION
# ==========================================================


def analyze_droplet_distribution():


    print("=" * 60)

    print(
        "Droplet Distribution Analysis"
    )

    print("=" * 60)


    label_files = glob.glob(
        "dataset/labels/**/*.txt",
        recursive=True
    )


    counts = []


    for file in label_files:


        with open(
            file,
            "r"
        ) as f:

            count = len(
                f.readlines()
            )


        counts.append(
            count
        )


    df = pd.DataFrame(

        {
            "image":
                [
                    Path(x).stem
                    for x in label_files
                ],

            "droplet_count":
                counts
        }

    )


    print(
        df.describe()
    )


    output = (
        "droplet_distribution.csv"
    )


    df.to_csv(
        output,
        index=False
    )


    print(
        f"\nSaved: {output}"
    )



    plt.figure(
        figsize=(8,5)
    )


    plt.hist(
        counts,
        bins=30
    )


    plt.xlabel(
        "Number of Droplets"
    )


    plt.ylabel(
        "Number of Images"
    )


    plt.title(
        "Droplet Distribution"
    )


    plt.show()



# ==========================================================
# VISUALIZE RANDOM LABELS
# ==========================================================


def visualize_dataset_sample(
        image_path
):

    """
    Display image with YOLO polygons
    """

    image = cv2.imread(
        image_path
    )


    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    label_path = os.path.join(

        "dataset/labels",

        Path(
            image_path
        ).parent.name,

        Path(
            image_path
        ).stem + ".txt"

    )


    h,w,_ = image.shape



    if not os.path.exists(
        label_path
    ):

        print(
            "Label not found"
        )

        return



    plt.figure(
        figsize=(8,8)
    )


    plt.imshow(
        image
    )


    with open(
        label_path,
        "r"
    ) as f:


        labels = f.readlines()



    for label in labels:


        data = (
            label
            .strip()
            .split()
        )


        points = np.array(
            data[1:],
            dtype=float
        )


        points = points.reshape(
            -1,
            2
        )


        points[:,0] *= w
        points[:,1] *= h


        plt.plot(

            np.append(
                points[:,0],
                points[0,0]
            ),

            np.append(
                points[:,1],
                points[0,1]
            )

        )


    plt.title(
        "Dataset Annotation Preview"
    )

    plt.axis(
        "off"
    )

    plt.show()



# ==========================================================
# CREATE DATASET REPORT
# ==========================================================


def create_dataset_report():


    print("="*60)

    print(
        "Creating Dataset Report"
    )

    print("="*60)


    report = {


        "Training Images":
            len(
                glob.glob(
                    TRAIN_FOLDER+"/*.jpg"
                )
            ),


        "Validation Images":
            len(
                glob.glob(
                    VAL_FOLDER+"/*.jpg"
                )
            ),


        "Testing Images":
            len(
                glob.glob(
                    TEST_FOLDER+"/*.jpg"
                )
            ),


        "Model":
            MODEL_NAME,


        "Image Size":
            IMAGE_SIZE,


        "Batch":
            BATCH_SIZE,


        "GPU":
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"


    }


    df = pd.DataFrame(

        report.items(),

        columns=[
            "Parameter",
            "Value"
        ]

    )


    df.to_csv(
        "dataset_report.csv",
        index=False
    )


    print(df)

    print(
        "\nDataset report saved."
    )