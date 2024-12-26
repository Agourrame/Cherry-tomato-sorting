import numpy as np  # Add this line
from model import create_model
from preprocess import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load training data
train_folder = 'data/train'
images, labels_color, labels_diameter = load_data(train_folder)

# Encode color labels (categorical: ripe, semi-ripe, unripe)
color_encoder = LabelEncoder()
labels_color_encoded = color_encoder.fit_transform(labels_color)  # e.g., ['ripe', 'unripe'] -> [0, 1]
labels_color_categorical = to_categorical(labels_color_encoded)

# Encode diameter labels (continuous: small, medium, large)
diameter_map = {'small': 0, 'medium': 1, 'large': 2}
labels_diameter_encoded = np.array([diameter_map[d] for d in labels_diameter])

# Split into train and validation sets
X_train, X_val, y_train_color, y_val_color, y_train_dia, y_val_dia = train_test_split(
    images, labels_color_categorical, labels_diameter_encoded, test_size=0.2, random_state=42
)

# Train model
model = create_model()
history = model.fit(
    X_train, {'color_output': y_train_color, 'diameter_output': y_train_dia},
    validation_data=(X_val, {'color_output': y_val_color, 'diameter_output': y_val_dia}),
    epochs=20,
    batch_size=32
)

# Save the trained model
model.save('outputs/models/tomato_sorter.h5')
print("Model training complete. Saved to outputs/models/tomato_sorter.h5")
