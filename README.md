# Smart Parking Occupancy Detection

A computer vision system for classifying predefined parking spaces as **free** or **occupied** from images and video streams. The project evaluates CNN-based parking-space classification and pretrained YOLO11 vehicle detection with polygon-based Region of Interest (ROI) analysis.

The selected YOLO11-ROI pipeline is integrated into a PyQt5 desktop application for monitoring up to three camera sources in real time.

## Overview

Camera-based parking monitoring can observe multiple parking spaces without requiring a physical sensor for every location. This project investigates two occupancy detection approaches:

| Approach                 | Description                                                                    | Primary Use                                   |
| ------------------------ | ------------------------------------------------------------------------------ | --------------------------------------------- |
| CNN classification       | Classifies cropped images of individual parking spaces as free or occupied     | Model comparison and occupancy classification |
| YOLO11 with ROI analysis | Detects vehicles and associates them with predefined polygonal parking regions | Image, video, and multi-camera monitoring     |

The project includes dataset preparation, model training, cross-validation, external evaluation, ROI configuration, occupancy analysis, and desktop application development.

## Key Features

* Preparation of CNRPark+EXT and PKLot datasets
* CNN training for binary parking occupancy classification
* Five-fold cross-validation and independent test-set evaluation
* External-image evaluation of the final CNN model
* Vehicle detection using pretrained YOLO11
* Manual polygon-based parking ROI configuration
* Parking occupancy analysis for images and videos
* Comparison against manually labeled ground truth
* PyQt5 desktop application supporting three camera sources
* Per-camera and system-wide occupancy statistics
* Real-time display of free and occupied parking spaces

## Occupancy Decision Logic

Each parking space is represented by a polygonal ROI. After vehicle detection, a parking space is classified as occupied when at least one of the following conditions is satisfied:

1. The bottom-center point of a detected vehicle lies inside the parking ROI.
2. The overlap between the vehicle bounding box and the parking ROI reaches the configured threshold.

If neither condition is satisfied, the parking space is classified as free.

## Repository Structure

```text
smart-parking-occupancy-detection/
|-- app/
|   `-- app_newversion.py
|-- demo/
|   |-- config/
|   |   |-- cam1_slots.json
|   |   |-- cam2_slots.json
|   |   `-- cam3_slots.json
|   `-- output/
|       |-- iu_result/
|       `-- web_result/
|-- src/
|   |-- data/
|   |   |-- check_dataset.py
|   |   |-- parking_dataset.py
|   |   |-- prepare_dataset.py
|   |   `-- prepare_pklot.py
|   |-- evaluation/
|   |   |-- analyze_5fold_results.py
|   |   `-- test_final_cnn_external.py
|   |-- inference/
|   |   `-- cnn_classifier.py
|   |-- models/
|   |   `-- simple_cnn.py
|   |-- training/
|   |   |-- 5fold_with_test.py
|   |   `-- train_final_cnn.py
|   `-- yolo_detect/
|       |-- demo_parking_roi_v3.py
|       `-- setup_roi_folder.py
|-- crop_cnn_external.py
|-- evaluate_manual_vs_model.py
|-- requirements.txt
`-- README.md
```

Large datasets, trained model weights, videos, virtual environments, and generated experiment outputs are excluded from the repository.

## Datasets

| Dataset                         | Usage                                            |
| ------------------------------- | ------------------------------------------------ |
| CNRPark+EXT                     | CNN-based parking-space classification           |
| PKLot                           | Comparative training and experimental evaluation |
| Locally recorded parking videos | Multi-camera application demonstration           |
| External parking images         | Qualitative model evaluation                     |

After dataset preparation, the expected CSV files are stored at:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Dataset files must be downloaded and prepared separately before model training.

## Technology Stack

* Python 3.11
* PyTorch
* Torchvision
* Ultralytics YOLO
* OpenCV
* PyQt5
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Pillow

## System Requirements

* Python 3.11
* Windows 10 or Windows 11
* A CUDA-compatible NVIDIA GPU is recommended for faster inference and training
* Sufficient storage for the CNRPark+EXT and PKLot datasets

The project can run on a CPU, but model training and multi-camera inference may be considerably slower.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/phucbaotran/smart-parking-occupancy-detection.git
cd smart-parking-occupancy-detection
```

### 2. Create a virtual environment

```powershell
py -3.11 -m venv venv_gpu
```

### 3. Activate the environment

```powershell
.\venv_gpu\Scripts\Activate.ps1
```

### 4. Install the dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Local Model Files

Pretrained and trained model weights are not stored in the repository because of their file sizes.

Before running the corresponding pipelines, download or provide the required model files locally, such as:

```text
yolo11n.pt
yolo11s.pt
final_cnrpark_cnn.pth
```

Model paths and input sources may need to be configured based on the local project directory.

## Usage

Run all commands from the repository root.

### Prepare the CNN dataset

```powershell
python src/data/prepare_dataset.py
```

### Run five-fold cross-validation

```powershell
python src/training/5fold_with_test.py
```

### Train the final CNN model

```powershell
python src/training/train_final_cnn.py
```

### Evaluate the final CNN on external images

```powershell
python src/evaluation/test_final_cnn_external.py
```

### Configure parking-space ROIs

```powershell
python src/yolo_detect/setup_roi_folder.py
```

The ROI configuration tool allows parking spaces to be defined manually as polygons and saved as JSON files.

### Run YOLO11 with ROI analysis

```powershell
python src/yolo_detect/demo_parking_roi_v3.py
```

### Compare predictions with manual ground truth

```powershell
python evaluate_manual_vs_model.py
```

### Launch the multi-camera application

```powershell
python app/app_newversion.py
```

Before starting the application, configure the required video sources, model weights, and ROI JSON files for each camera.

## Example Outputs

### Locally collected parking image

![YOLO11 and ROI result on a locally collected parking image](demo/output/iu_result/iu_01_roi_detected.jpg)

### External parking image

![YOLO11 and ROI result on an external parking image](demo/output/web_result/web_01_roi_detected.jpg)

## Limitations

* Parking-space ROIs must be configured manually.
* The ROI positions depend on a mostly fixed camera viewpoint.
* Camera movement can cause ROI misalignment and reduce occupancy accuracy.
* Small, distant, or partially occluded vehicles may not be detected reliably.
* Objects with vehicle-like visual features may occasionally produce false detections.
* Lighting, shadows, weather, viewing angle, and video quality can affect system performance.
* Large datasets, videos, and trained model weights are not distributed with the repository.

## Future Improvements

* Automatic parking-space localization
* Camera stabilization and ROI alignment
* Improved performance under nighttime and difficult weather conditions
* Faster inference for live video streams
* Support for additional cameras and IP camera sources
* Persistent storage of parking occupancy history
* Automated notifications and parking availability reporting
* Deployment as a web-based or cloud-connected monitoring system

## Author

**Tran Bao Phuc**
Electrical and Telecommunication Engineering Student

This repository was developed as an individual senior project and technical portfolio project focusing on deep learning, object detection, and computer vision.
