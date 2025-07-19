# ASL Detector – Real-Time American Sign Language Recognition

*ASL Detector* is an intelligent, real-time hand gesture recognition system that deciphers *American Sign Language (ASL)* alphabets using the magic of computer vision and machine learning. With just a webcam and your hands, you can bridge silence and communication — one letter at a time.

## Overview

This project uses *MediaPipe* to detect 21 hand landmarks and a *Multi-Layer Perceptron (MLP)* classifier to recognize ASL alphabets (A–Z). It unfolds in three core phases:

1. *Data Collection* – Capture hand landmark data and store it as CSV files.
2. *Model Training* – Train a neural network (MLP) on the landmark dataset.
3. *Real-Time Prediction* – Predict ASL letters live from webcam input.

## Features

- *Real-time* ASL letter detection  
- *21-point* hand landmark tracking using MediaPipe  
- *A–Z alphabet classification*  
- *Lightweight* and optimized to run on CPU  
- Fully *modular*, easy to extend or adapt  

## Tech Stack
| Tool                            | Purpose                                                |                      
|------------------------ |-----------------------------------------  |
| *Python*                 | Core programming language              |
| *OpenCV*               | Webcam handling and visualization    |
| *MediaPipe*           | Hand landmark detection                    |
| *Scikit-learn*          | MLPClassifier for model training          |
| *Pandas / NumPy* | Data preprocessing and manipulation |

## Project Structure

ASL-Detection/
├── data_collection.py # Script to record hand gesture data
├── model_training.py # Script to train MLPClassifier on collected data
├── asl_detector.py # Real-time ASL recognition from webcam
└── README.md 

## Clone the repository and install the dependencies:

```bash
git clone https://github.com/SnehaSinghwe/ASL detection.git
cd ASL detection

## Collect Hand Gesture Data
python collector.py

## Train the Model
python train.py

## Run Real-Time Detector
python detect.py
