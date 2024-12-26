from tensorflow.keras import models, layers

def create_model(input_shape=(128, 128, 3)):
    input_layer = layers.Input(shape=input_shape)

    # Shared feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    # Color classification head
    color_output = layers.Dense(3, activation='softmax', name='color_output')(x)

    # Diameter regression head
    diameter_output = layers.Dense(1, activation='linear', name='diameter_output')(x)

    # Build and compile model
    model = models.Model(inputs=input_layer, outputs=[color_output, diameter_output])
    model.compile(
        optimizer='adam',
        loss={'color_output': 'sparse_categorical_crossentropy', 'diameter_output': 'mse'},
        metrics={'color_output': 'accuracy', 'diameter_output': 'mae'}
    )
    return model
