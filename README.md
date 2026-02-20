# Digit Recognition System
This project is a machine learning application designed to recognize handwritten digits using Convolutional Neural Networks (CNN). It includes tools for data preprocessing, model training, and an interactive drawing interface for real-time predictions.

## Data source
- **Source**: [MNIST in CSV (Kaggle)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **Format**: Tabular CSV containing grayscale pixel values (0–255).
- **Target**: The label column representing the digit (0–9).
- **Dimensions**: Aside from the label, each row consists of 784 features (a flattened 28×28 pixel image).

## Key Features
- **Data processing for model training**: Processes raw CSV data into $28 \times 28$ grayscale images by parsing pixel 
  coordinates (from train_mnist.csv).
- **Dynamic Data Augmentation (new)**: Enhances model generalization using random rotations, translations, and zooms.
- **Automated Hyperparameter Tuning (new)**: Utilizes BayesianOptimization to search for the best architecture, including the number of layers, filters, dropout rates, and optimizers.
- **Comprehensive Evaluation**: Features detailed visualization of label distribution, confusion matrices, and an analysis of misclassified samples.
- **Live Drawing Canvas**: Includes a built-in GUI that allows users to draw digits with a mouse and get instant predictions from the trained model.

## Tech Stack
- **Core**: Python, TensorFlow/Keras
- **Tuning**: Keras Tuner
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib
- **GUI & Imaging**: Tkinter, Pillow (PIL)

## Results
Based on the tuning process, the model typically achieves **>98%** accuracy on the test set.
