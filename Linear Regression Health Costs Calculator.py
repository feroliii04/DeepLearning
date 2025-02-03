import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

import requests

url = "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"
response = requests.get(url)

# Verificar que la respuesta sea correcta
if response.status_code == 200:
    with open("insurance.csv", "wb") as file:
        file.write(response.content)
    print("Archivo descargado correctamente")
else:
    print(f"Error al descargar el archivo: {response.status_code}")

dataset = pd.read_csv('insurance.csv')
print(dataset.tail())

# Convert categorical columns to numerical values
dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'], drop_first=True)

# Split features and labels
labels = dataset.pop('expenses')

# Split the data into training (80%) and testing (20%)
from sklearn.model_selection import train_test_split
train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    dataset, labels, test_size=0.2, random_state=42
)

# Normalize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_dataset = scaler.fit_transform(train_dataset)
test_dataset = scaler.transform(test_dataset)

# Build the model
def build_model():
    model = keras.Sequential([
        layers.Input(shape=[train_dataset.shape[1]]),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['mae', 'mse']
    )
    return model

model = build_model()

# Train the model
epochs = 200
history = model.fit(
    train_dataset, train_labels,
    epochs=epochs,
    validation_split=0.2,
    verbose=1,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
)


# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("You passed the challenge. Great job!")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_dataset).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
