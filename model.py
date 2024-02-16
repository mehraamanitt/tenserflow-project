import tensorflow as tf
from tensorflow.keras import layers, models

# Example dataset (dummy data)
X_train = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
y_train = [0, 1, 0]

# Define the model architecture
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1)

# Example of making predictions
X_test = [[0.4, 0.5], [0.5, 0.6]]
predictions = model.predict(X_test)
print("Predictions:", predictions)
