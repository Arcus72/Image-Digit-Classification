# Digit Classification System
This project is a machine learning application designed to recognize handwritten digits using Convolutional Neural 
Networks (CNN). It includes tools for data preprocessing, model training, and an interactive drawing interface for real-time predictions.

👉 [Live Demo](https://huggingface.co/spaces/Arcus72/Image-Digit-Classification)

## Files
- **app.py**: The main application file that creates a Gradio interface for the digit recognition system.
- **model_training.ipynb**: A Jupyter notebook that contains the data processing, model training, and evaluation code.
- **mnist_dataset.csv**: The dataset used for training the model.

## Data source
- **Source**: [MNIST in CSV (Kaggle)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **Format**: CSV containing grayscale pixel values (0–255).
- **Target**: The label column representing the digit (0–9).
- **Dimensions**: Aside from the label, each row consists of 784 features (a flattened 28×28 pixel image).

## Key Features
- **Data processing for model training**: Processes raw CSV data into $28 \times 28$ grayscale images by parsing pixel 
  coordinates (from train_mnist.csv).
- **Comprehensive Evaluation**: Features detailed visualization of label distribution, confusion matrices, and an analysis of misclassified samples.
- **Interactive Demo**: A Gradio interface that allows users to draw digits and get instant predictions from the trained model.

## Tech Stack
- **Core**: Python, Pytorch. (TensorFlow/Keras version on main branch)
- **Deep Learning Frameworks**:
  - **PyTorch version** - pytorch-version branch (this branch)
  - **TensorFlow/Keras** - main branch
- **Tuning**: Keras Tuner
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Tkinter, Pillow (PIL)
- **UI**: Gradio

## Upcoming Changes:
- **Improve demo** - The model currently struggles to identify certain handwritten digits in demo. Potential solution - Aligning demo input 
  styles with training data for improved model's accuracy. (The pen used in demo is not the same as in data set).
- **Dynamic Data Augmentation**.
- **Automated Hyperparameter Tuning**.

## Results
Based on the tuning process, the model typically achieves **>97%** accuracy on the test set.
