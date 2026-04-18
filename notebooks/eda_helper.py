import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_distribution(labels):
    """
    Plots the distribution of classes.
    """
    sns.countplot(labels)
    plt.title("Class Distribution: Benign vs Malignant")
    plt.show()

def show_samples(images, labels, cols=4):
    """
    Displays a grid of sample images with labels.
    """
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
