# CIFAR-10 Image Classification Project

This project implements a CIFAR-10 image classification pipeline using TensorFlow and Keras, with a PyQt6-based graphical user interface (GUI) for real-time image classification. The model is trained on the CIFAR-10 dataset, and predictions can be made via a GUI with drag-and-drop or file selection functionality.

---

## Project Structure

```
project_folder/
├── data/
│   └── cifar10/
│       └── cifar-10-batches/    # CIFAR-10 dataset files
├── src/
│   ├── model.py                # Script to train and save the CIFAR-10 model
│   ├── testing.py              # Script to evaluate the model and display predictions
│   ├── gui.py                  # PyQt6-based GUI for real-time image classification
├── venv/                       # Virtual environment (recommended for dependencies)
├── requirements.txt            # Python dependencies for the project
└── README.md                   # Project documentation

```


## Getting Started

### 1. Install Dependencies

It is recommended to use a virtual environment to manage dependencies. Install the required Python libraries using `requirements.txt`:

# Create and activate a virtual environment
'''bash
python -m venv venv
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate          # On Windows

# Install dependencies
pip install -r requirements.txt


2. Download the CIFAR-10 Dataset
Place the CIFAR-10 dataset (e.g., cifar-10-batches-py) inside the data/cifar10/cifar-10-batches/ directory. The dataset should be in its original binary format, as expected by the scripts.

3. Train the Model
Run the model.py script to train the CNN model on the CIFAR-10 dataset.
python src/model.py

This script:

- Loads and preprocesses the CIFAR-10 dataset.
- Trains a Convolutional Neural Network (CNN) model.
- Saves the trained model to `data/model.keras`.


#4. Test the Model

Run the `testing.py` script to evaluate the trained model and display predictions on randomly selected test images:
'''bash
python src/testing.py


5. Run the GUI
Run the gui.py script to launch a PyQt6-based graphical interface for image classification.

python src/gui.py
The GUI allows users to:

Drag and drop an image file for classification.
Select an image file via a file dialog.
Scripts Overview
1. model.py
Trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.
Visualizes training and validation accuracy/loss.
Saves the trained model to data/model.keras.
2. testing.py
Loads the pre-trained model and CIFAR-10 test data.
Makes predictions on the test dataset.
Displays random test images with predicted class labels.
3. gui.py
PyQt6-based GUI for real-time image classification.
Features:
Drag-and-drop support for images.
File dialog for selecting images.
Displays predictions and the input image.
Dependencies
The project dependencies are listed in requirements.txt. Install them using the following command:

bash
Copy
Edit
pip install -r requirements.txt
Key Dependencies:
Python 3.8+
TensorFlow/Keras
Numpy
Matplotlib
PyQt6
Pillow
Dataset
The project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
Place the CIFAR-10 dataset in data/cifar10/cifar-10-batches/ before running the scripts.

Usage Examples
Train the Model
Run model.py to train and save the model:

bash
Copy
Edit
python src/model.py
Test the Model
Run testing.py to evaluate the model:

bash
Copy
Edit
python src/testing.py
Classify Images via GUI
Run gui.py to classify images using the GUI:

bash
Copy
Edit
python src/gui.py
Output
Model Training: Plots showing training and validation accuracy/loss during training.
Testing Script: A grid of test images with predicted class labels.
GUI: Real-time predictions displayed on selected or dropped images.
Acknowledgments
The project uses the CIFAR-10 dataset provided by the Canadian Institute For Advanced Research.
Developed using TensorFlow, Keras, and PyQt6.
