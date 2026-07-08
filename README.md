# 🚗 Smart Parking Occupancy Detection

A deep learning project for parking occupancy classification using PyTorch and Computer Vision.
The project aims to provide a lightweight alternative to sensor-based smart parking systems by classifying parking slots directly from camera images. It includes dataset preparation, model training, cross-validation, evaluation, and is being extended toward real-time deployment.

## Features
- CNN-based parking occupancy classification
- PyTorch training pipeline
- 5-fold cross-validation
- Evaluation on unseen test data
- Modular project structure
- Ongoing Streamlit deployment

## Dataset
- Dataset: CNRPark+EXT
- Images: 144,965
- Occupied: 79,281
- Vacant: 65,684

## Results

| Metric    | Score |
|---------  |------:|
| Accuracy  | 97.43% |
| Precision | 98.69% |
| Recall    | 96.59% |
| F1-score  | 97.63% |

## Tech Stack
Python
PyTorch
Torchvision
OpenCV
NumPy
Pandas
Scikit-learn
Matplotlib
Streamlit

## Installation
git clone ...
cd ...
pip install -r requirements.txt

## Usage
python train.py
python evaluate.py
python train.py
python evaluate.py

## Roadmap
- [x] CNN training pipeline
- [x] Custom Dataset
- [x] Cross-validation
- [x] Model evaluation
- [ ] Streamlit dashboard
- [ ] ROI extraction
- [ ] Real-time CCTV inference
- [ ] YOLO-based parking detection

## About

Hi, I'm Tran Bao Phuc.

I'm a final-year Electrical and Telecommunication Engineering student interested in Computer Vision, Machine Learning, and Data Science. I build projects to strengthen my practical skills and explore real-world AI applications.
