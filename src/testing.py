import numpy as np
import os
import pickle
from tensorflow.keras import models
import matplotlib.pyplot as plt


def load_cifar10_data(data_dir):
    """
    Load CIFAR-10 test data from a given directory.

    Args:
        data_dir (str): Path to CIFAR-10 dataset directory.

    Returns:
        Tuple: Testing data as (x_test, y_test).
    """
    # Construct the path to the test batch dynamically
    test_batch_path = os.path.join(data_dir, 'test_batch')
    
    with open(test_batch_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
        x_test = test_batch['data']
        y_test = test_batch['labels']

    # Reshape and normalize the data
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0

    return x_test, np.array(y_test)


if __name__ == "__main__":
    # Define base directory dynamically
    base_dir = os.path.abspath("../data/cifar10/cifar-10-batches-py")
    model_path = os.path.abspath("../data/model.keras")

    # Load test dataset
    x_test, y_test = load_cifar10_data(base_dir)

    # Load the pre-trained model
    model = models.load_model(model_path)

    # Predict on the test dataset
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Display random test images with predicted labels
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_images = 16
    indices = np.random.choice(len(x_test), num_images, replace=False)

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_test[idx])
        plt.xlabel(class_names[predicted_classes[idx]])
    plt.show()
