import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt  # Import for plotting

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

# Load the IMDb dataset
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# Preprocess the data
train_data = train_data.shuffle(10000).batch(512).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_data = validation_data.batch(512).prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.batch(512).prefetch(buffer_size=tf.data.AUTOTUNE)

# Load the pre-trained embedding layer from TensorFlow Hub
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

# Build the model
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Use sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data,
                    epochs=10,
                    validation_data=validation_data,
                    verbose=1)

# Evaluate the model on the test set
results = model.evaluate(test_data, verbose=2)

# Print the evaluation results
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
