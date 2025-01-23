import sys
import numpy as np
from tensorflow.keras import models
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PIL import Image
from PyQt6.QtCore import Qt


class DropImageLabel(QLabel):
    """
    QLabel subclass that supports drag-and-drop functionality for images.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.setText("Drag and Drop an Image Here or Choose an Image")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 2px dashed gray;")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.parent().classify_image(file_path)
            else:
                self.parent().prediction_label.setText("Invalid file type. Please drop an image file.")
        else:
            self.parent().prediction_label.setText("No file detected.")


class ImageClassifierApp(QWidget):
    """
    PyQt6-based GUI for CIFAR-10 image classification.
    """
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = self.load_model()

    def initUI(self):
        """
        Initialize the GUI layout and widgets.
        """
        self.setWindowTitle('CIFAR-10 Image Classifier')
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        # Drag-and-drop label
        self.label = DropImageLabel(self)
        self.layout.addWidget(self.label)

        # Button to select an image
        self.button = QPushButton('Choose Image', self)
        self.button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.button)

        # Label for predictions
        self.prediction_label = QLabel('', self)
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.prediction_label)

        self.setLayout(self.layout)

    def load_model(self):
        """
        Load the pre-trained model for CIFAR-10 classification.
        """
        model_path = '../data/model.keras'
        return models.load_model(model_path)

    def open_file_dialog(self):
        """
        Open a file dialog to select an image.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)")
        if file_path:
            self.classify_image(file_path)

    def classify_image(self, file_path):
        """
        Classify the selected or dropped image using the pre-trained model.
        """
        try:
            image = Image.open(file_path).resize((32, 32))
            image_array = np.array(image) / 255.0

            if image_array.ndim == 3 and image_array.shape[2] == 3:
                image_array = image_array.reshape(1, 32, 32, 3)
                predictions = self.model.predict(image_array)
                predicted_class = np.argmax(predictions, axis=1)[0]
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                predicted_label = class_names[predicted_class]

                self.prediction_label.setText(f'Predicted: {predicted_label}')
                self.display_image(file_path)
            else:
                self.prediction_label.setText('Please select a color image.')
        except Exception as e:
            self.prediction_label.setText(f'Error: {str(e)}')

    def display_image(self, file_path):
        """
        Display the selected or dropped image in the label.
        """
        pixmap = QtGui.QPixmap(file_path)
        self.label.setPixmap(pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec())
