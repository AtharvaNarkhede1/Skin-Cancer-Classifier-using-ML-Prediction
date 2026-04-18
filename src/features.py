import cv2
import numpy as np

def extract_color_distribution(image):
    """
    Extracts mean and standard deviation for R, G, B channels.
    """
    means = np.mean(image, axis=(0, 1))
    stds = np.std(image, axis=(0, 1))
    return np.concatenate([means, stds])

def estimate_asymmetry(image):
    """
    A simple estimation of asymmetry by comparing the image with its flipped versions.
    """
    gray = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    
    # Horizontal flip
    h_flip = cv2.flip(gray, 1)
    h_diff = np.sum(cv2.absdiff(gray, h_flip))
    
    # Vertical flip
    v_flip = cv2.flip(gray, 0)
    v_diff = np.sum(cv2.absdiff(gray, v_flip))
    
    return np.array([h_diff, v_diff])

def extract_features(image):
    """
    Combines various features into a single vector.
    """
    color = extract_color_distribution(image)
    asymmetry = estimate_asymmetry(image)
    
    # Combine all features
    features = np.concatenate([color, asymmetry])
    return features

if __name__ == "__main__":
    print("Feature extraction module ready.")
