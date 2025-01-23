import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


def load_cifar10_data(data_dir):
    """
    Load CIFAR-10 data from a given directory.

    Args:
        data_dir (str): Path to CIFAR-10 dataset directory.

    Returns:
        Tuple: Training and testing data as (x_train, y_train), (x_test, y_test).
    """
    # Load the training data
    x_train, y_train = [], []
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            x_train.append(batch['data'])
            y_train.append(batch['labels'])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    # Load the test data
    test_batch_path = os.path.join(data_dir, "test_batch")
    with open(test_batch_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
        x_test = test_batch['data']
        y_test = test_batch['labels']

    # Reshape and normalize the data
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0

    return (x_train, np.array(y_train)), (x_test, np.array(y_test))


def plot_history(history):
    """
    Plot training and validation accuracy and loss.

    Args:
        history: Keras model training history object.
    """
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.show()


if __name__ == "__main__":
    # Define paths dynamically
    base_dir = os.path.abspath("../data/cifar10/cifar-10-batches-py")
    model_save_path = os.path.abspath("../data/model.keras")

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(base_dir)

    # Define class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Build the CNN model
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Plot training history
    plot_history(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Save the trained model
    model.save(model_save_path)
