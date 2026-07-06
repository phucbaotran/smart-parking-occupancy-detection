\# Smart Parking Occupancy Detection



A deep learning and computer vision project for parking slot occupancy classification.



This project focuses on detecting whether a parking slot is \*\*Empty\*\* or \*\*Occupied\*\* from image data. The current implementation provides a complete image classification pipeline, including dataset preparation, custom PyTorch dataset loading, CNN model training, K-Fold cross-validation, and model evaluation.



\---



\## Project Overview



Finding available parking spaces is a common problem in urban areas, commercial buildings, universities, and public parking lots. Traditional smart parking systems often rely on physical sensors installed in each parking slot, which can increase hardware cost and maintenance complexity.



This project uses \*\*computer vision\*\* as an alternative approach. By using parking slot images, the system learns to classify the occupancy status of each slot without requiring one physical sensor per parking space.



The current version focuses on the \*\*parking slot image classification stage\*\*. Each input image represents one parking slot, and the model predicts whether that slot is empty or occupied.



\---



\## Problem Definition



This project is formulated as a binary image classification task.



```text

Input  : Parking slot image

Output : Empty or Occupied

```



The model learns visual features from labeled parking slot images and predicts the occupancy status of unseen samples.



\---



\## Objectives



The main objectives of this project are:



\- Prepare and organize labeled parking occupancy data

\- Build a reusable PyTorch dataset class

\- Train a CNN model for binary parking slot classification

\- Evaluate model performance using classification metrics

\- Apply K-Fold cross-validation to check model stability

\- Maintain a clean and reproducible project structure



\---



\## Why Computer Vision for Smart Parking?



Computer vision is suitable for smart parking systems because cameras can monitor multiple parking slots at the same time. Compared with sensor-based systems, image-based solutions may reduce hardware installation cost and can potentially reuse existing CCTV or surveillance cameras.



Advantages of using computer vision include:



\- Camera-based monitoring of parking areas

\- Lower dependency on physical sensors

\- Better scalability for large parking lots

\- Visual information for parking status analysis

\- Compatibility with deep learning models



However, computer vision systems may still be affected by:



\- Lighting changes

\- Shadows

\- Camera angle differences

\- Occlusion between vehicles

\- Weather conditions

\- Nighttime image quality



\---



\## Current Project Scope



The current implementation includes:



\- Dataset checking

\- Dataset preparation

\- Custom PyTorch dataset class

\- CNN-based occupancy classification

\- Training and validation pipeline

\- K-Fold cross-validation

\- Test set evaluation



The current version does not yet include:



\- Real-time video processing

\- Automatic parking slot localization

\- Web dashboard

\- Full production deployment



\---



\## Project Structure



```text

smart-parking-occupancy-detection/

├── src/

│   ├── data/

│   │   ├── check\_dataset.py

│   │   ├── parking\_dataset.py

│   │   └── prepare\_dataset.py

│   │

│   ├── training/

│   │   ├── train.py

│   │   └── cross\_validation.py

│   │

│   └── evaluation/

│       └── evaluate.py

│

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\## File Description



\### `src/data/check\_dataset.py`



Checks the dataset before training.



Main purposes:



\- Verify dataset paths

\- Count dataset samples

\- Check class distribution

\- Detect missing or invalid image paths



\---



\### `src/data/prepare\_dataset.py`



Prepares the dataset for training, validation, and testing.



Main purposes:



\- Read image paths

\- Assign labels

\- Split data into train, validation, and test sets

\- Save processed dataset files



\---



\### `src/data/parking\_dataset.py`



Defines the custom PyTorch dataset class.



Main purposes:



\- Load image paths and labels

\- Read parking slot images

\- Apply image transformations

\- Return image tensors and labels for model training



\---



\### `src/training/train.py`



Trains the CNN model.



Main purposes:



\- Load training and validation data

\- Define the CNN model

\- Set loss function and optimizer

\- Train the model across epochs

\- Validate the model

\- Save the best-performing model



\---



\### `src/training/cross\_validation.py`



Performs K-Fold cross-validation.



Main purposes:



\- Split the dataset into multiple folds

\- Train and validate the model on different data splits

\- Evaluate model stability and generalization



\---



\### `src/evaluation/evaluate.py`



Evaluates the trained model on the test dataset.



Main purposes:



\- Load the trained model

\- Run prediction on test data

\- Calculate evaluation metrics

\- Report classification performance



\---



\## Dataset



This project is designed to work with labeled parking occupancy datasets such as:



\- CNRPark

\- CNRPark-EXT

\- PKLot

\- Custom parking slot image datasets



Expected classes:



```text

empty

occupied

```



Example processed dataset files:



```text

data/processed/train.csv

data/processed/val.csv

data/processed/test.csv

```



The dataset is not included in this repository because image datasets are usually large. Users need to prepare or download the dataset separately.



\---



\## Methodology



The project follows this pipeline:



```text

Dataset Collection

&#x20;       ↓

Dataset Preparation

&#x20;       ↓

Image Preprocessing

&#x20;       ↓

CNN Model Training

&#x20;       ↓

Validation

&#x20;       ↓

K-Fold Cross-Validation

&#x20;       ↓

Model Evaluation

```



The CNN model is trained to classify each parking slot image into one of two categories: empty or occupied.



\---



\## Model Approach



This project uses a Convolutional Neural Network for binary image classification.



CNNs are suitable for this task because they can learn visual patterns from images, such as:



\- Vehicle shape

\- Parking slot texture

\- Shadow and background patterns

\- Visual differences between empty and occupied spaces



The model outputs a binary prediction:



```text

0 = Empty

1 = Occupied

```



\---



\## Evaluation Metrics



The model is evaluated using the following metrics:



\- Accuracy

\- Precision

\- Recall

\- F1-score



These metrics are used to measure the classification performance of the model on validation and test data.



\### Accuracy



Measures the percentage of correct predictions.



\### Precision



Measures how many predicted occupied samples are actually occupied.



\### Recall



Measures how many actual occupied samples are correctly detected.



\### F1-score



Combines precision and recall into one balanced metric.



\---



\## Technologies Used



\- Python

\- PyTorch

\- Torchvision

\- OpenCV

\- NumPy

\- Pandas

\- Scikit-learn

\- Matplotlib

\- TQDM

\- Pillow



\---



\## Installation



Clone the repository:



```bash

git clone https://github.com/phucbaotran/smart-parking-occupancy-detection.git

```



Move into the project folder:



```bash

cd smart-parking-occupancy-detection

```



Create a virtual environment:



```bash

py -3.11 -m venv venv

```



Activate the virtual environment on Windows:



```bash

venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\## How to Run



Check the dataset:



```bash

python src/data/check\_dataset.py

```



Prepare the dataset:



```bash

python src/data/prepare\_dataset.py

```



Train the model:



```bash

python src/training/train.py

```



Run K-Fold cross-validation:



```bash

python src/training/cross\_validation.py

```



Evaluate the trained model:



```bash

python src/evaluation/evaluate.py

```



\---



\## Current Status



Completed:



\- Dataset preparation pipeline

\- Custom PyTorch dataset class

\- CNN training script

\- Cross-validation script

\- Evaluation script

\- GitHub repository setup



Planned improvements:



\- Add confusion matrix visualization

\- Add training and validation plots

\- Add sample prediction results

\- Add model architecture summary

\- Improve CNN architecture

\- Add real-time video processing

\- Add parking slot ROI extraction

\- Add simple dashboard or user interface



\---



\## Limitations



The current project has the following limitations:



\- Dataset files are not included in the repository

\- Trained model weights are not included

\- The current implementation focuses on image classification only

\- Real-time video processing is not implemented yet

\- Parking slot localization is not automated yet

\- Model performance may depend on lighting conditions, camera angle, shadows, and dataset quality



\---



\## Future Development



Future versions of this project may include:



\- Real-time parking occupancy detection from video

\- Parking slot ROI management

\- Vehicle detection integration using object detection models

\- Web-based dashboard for displaying parking status

\- Deployment using Streamlit, Flask, or FastAPI

\- Testing on custom parking lot images

\- Model optimization for faster inference



\---



\## Author



\*\*Tran Bao Phuc\*\*



Electrical and Telecommunication Engineering Student  

Interested in Artificial Intelligence, Machine Learning, and Computer Vision



\---



\## Repository Purpose



This repository is maintained as an independent computer vision project for learning, experimentation, and portfolio development.



The project demonstrates practical skills in:



\- Deep learning

\- Image classification

\- Computer vision

\- PyTorch model development

\- Dataset preparation

\- Model evaluation

\- Python project organization

