import cv2
import os
import numpy as np

def preprocess_image(image_path, target_size=(128, 128)):
    """Resize and normalize the image."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image / 255.0

def extract_labels_from_filename(filename):
    """
    Extract color (ripe, semi-ripe, unripe) and diameter (small, medium, large)
    from the filename.
    Example: "ripe_small_1.jpg" -> ("ripe", "small")
    """
    try:
        parts = filename.split('_')
        color = parts[0]  # ripe, unripe, semi
        diameter = parts[1]  # small, medium, large
        
        # Validate extracted labels
        if color not in {'ripe', 'semi-ripe', 'unripe'}:
            raise ValueError(f"Unexpected color label in filename: {filename}")
        if diameter not in {'small', 'medium', 'large'}:
            raise ValueError(f"Unexpected diameter label in filename: {filename}")
        
        return color, diameter
    except (IndexError, ValueError) as e:
        print(f"Error extracting labels from filename: {filename} -> {e}")
        return None, None

def load_data(folder_path):
    """
    Load images and their labels from a directory.
    Assumes filenames include labels in the format: color_diameter_X.jpg
    """
    images = []
    labels_color = []
    labels_diameter = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder_path, filename)
            color, diameter = extract_labels_from_filename(filename)
            
            if color is None or diameter is None:
                continue  # Skip invalid files
            
            images.append(preprocess_image(image_path))
            labels_color.append(color)
            labels_diameter.append(diameter)

    return np.array(images), np.array(labels_color), np.array(labels_diameter)
