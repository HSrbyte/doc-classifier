import matplotlib.pyplot as plt

from typing import Tuple


def plot_history(history: dict, figsize: Tuple[int, int] = (20, 5)) -> None:
    """
    Plots the training and validation accuracy and loss from a training history.

    Args:
        history (dict): A dictionary containing the training history with keys 'accuracy', 'val_accuracy', 'loss', and 'val_loss'.
        figsize (tuple, optional): The size of the figure for the plots. Defaults to (20, 5).

    Returns:
        None: This function does not return anything. It displays the plots.
    """
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(accuracy) + 1)

    # Créer une figure avec deux sous-graphes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Tracer les courbes de précision
    ax1.plot(epochs, accuracy, 'b', label='Training accuracy')
    ax1.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    ax1.set_title('Training and validation accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Tracer les courbes de perte
    ax2.plot(epochs, loss, 'b', label='Training loss')
    ax2.plot(epochs, val_loss, 'r', label='Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()

    # Afficher les graphiques
    plt.show()